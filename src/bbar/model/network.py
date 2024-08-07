import torch
import torch.nn as nn

from typing import Tuple, Dict, OrderedDict, Union, Optional
from torch import FloatTensor, LongTensor
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.typing import Adj
from bbar.utils.typing import NodeVector, EdgeVector, GraphVector, PropertyVector

from .layers import GraphEmbeddingModel, ConditionEmbeddingModel, PropertyPredictionModel,\
        TerminationPredictionModel, BlockSelectionModel, AtomSelectionModel
from bbar.transform import NUM_ATOM_FEATURES, NUM_BOND_FEATURES, NUM_BLOCK_FEATURES

class BlockConnectionPredictor(nn.Module) :
    def __init__ (self, cfg, property_information: OrderedDict[str, Tuple[float, float]]) :
        super(BlockConnectionPredictor, self).__init__()
        self._cfg = cfg
        # Property Information
        # {property_name: (mean, std)}
        self.property_information = property_information
        self.property_keys = property_information.keys()
        self.property_dim = len(property_information)

        # Graph Embedding
        self.core_graph_embedding_model = GraphEmbeddingModel(NUM_ATOM_FEATURES, NUM_BOND_FEATURES,
                                            0, **cfg.GraphEmbeddingModel_Core)

        self.block_graph_embedding_model = GraphEmbeddingModel(NUM_ATOM_FEATURES, NUM_BOND_FEATURES,
                                            NUM_BLOCK_FEATURES, **cfg.GraphEmbeddingModel_Block)

        # Property Regression
        self.property_prediction_model = PropertyPredictionModel(property_dim = self.property_dim, \
                                            **cfg.PropertyPredictionModel)

        # Condition Embedding
        self.condition_embedding_model = ConditionEmbeddingModel(condition_dim = self.property_dim, \
                                            **cfg.ConditionEmbeddingModel)

        # Terminate Prediction
        self.termination_prediction_model = TerminationPredictionModel(**cfg.TerminationPredictionModel)

        # Building Block Section
        self.block_selection_model = BlockSelectionModel(**cfg.BlockSelectionModel)

        # Atom Selection
        self.atom_selection_model = AtomSelectionModel(NUM_BOND_FEATURES,
                                            **cfg.AtomSelectionModel)

    def standardize_property(
        self,
        property: Dict[str, Union[float, FloatTensor]],
    ) -> PropertyVector :
        assert property.keys() == self.property_keys, \
            f"Input Keys is not valid\n" \
            f"\tInput:      {set(property.keys())}\n" \
            f"\tRequired:   {set(self.property_keys)}"

        property = [(property[key] - mean) / std
                                for key, (mean, std) in self.property_information.items()]
        if isinstance(property[0], float) :
            property = torch.FloatTensor([property])        # (1, Fc)
        else :
            property = torch.stack(property, dim=-1)        # (N, Fc)

        return property 
        
    def core_molecule_embedding(
        self,
        batch: Union[PyGData, PyGBatch],
    ) -> Tuple[NodeVector, GraphVector]:
        """
        Input:
            batch: PyGData or PyGBatch. (Transform by CoreGraphTransform) 
        Output:
            x_upd_core: Updated Node Vector             (V_core, Fh_core)
            Z_core: Graph Vector                        (N, Fz_core)
        """
        return self.core_graph_embedding_model.forward_batch(batch)

    def building_block_embedding(
        self,
        batch: Union[PyGData, PyGBatch]
    ) -> Tuple[NodeVector, GraphVector] :
        """
        Input:
            batch: PyGData or PyGBatch. (Transform by BlockGraphTransform) 
        Output: 
            x_upd_block: Updated Node Vector            (V_block, Fh_block)     # Unused
            Z_block: Graph Vector                       (N, Fz_block)           # Used
        """
        return self.block_graph_embedding_model.forward_batch(batch)

    def get_property_prediction(self, Z_core: GraphVector) -> PropertyVector:
        """
        Input:
            Z_core: Graph Vector                        (N, Fz_core)
        Output: 
            y_hat_property: PropertyVector                (N, Fc)
        """
        return self.property_prediction_model(Z_core)

    def condition_embedding(
        self,
        x_upd_core: NodeVector,
        Z_core: GraphVector,
        condition: PropertyVector,
        node2graph_core: Optional[LongTensor] = None,
    ) -> Tuple[NodeVector, GraphVector]:
        """
        Input:
            x_upd_core: Updated Node Vector             (V_core, Fh_core)
            Z_core: Graph Vector                        (N, Fz_core)
            condition: Condition Vector                 (N, Fc)
        Output:
            x_upd_core: Updated Node Vector             (V_core, Fh_core)
            Z_core: Graph Vector                        (N, Fz_core)
        """
        return self.condition_embedding_model(x_upd_core, Z_core, condition, node2graph_core)

    def get_termination_logit(self, Z_core: GraphVector) -> FloatTensor:
        """
        Input:
            Z_core: From core_molecule_embedding        (N, Fz_core)
        Output:
            p_term: Termination logit                   (N, )
        """
        return self.termination_prediction_model(Z_core, return_logit=True)
    
    def get_termination_probability(self, Z_core: GraphVector) -> FloatTensor:
        """
        Input:
            Z_core: From core_molecule_embedding        (N, Fz_core)
        Output:
            p_term: Termination Probability             (N, )
        """
        return self.termination_prediction_model(Z_core)

    def get_block_priority(self, Z_core: GraphVector, Z_block) -> FloatTensor:
        """
        Input:
            Z_core: From core_molecule_embedding        (N, Fz_core)
            Z_block: From building_block_embedding      (N, Fz_block)
        Output:
            p_block: Block Priority                     (N, )
        """
        return self.block_selection_model(Z_core, Z_block)

    def get_atom_probability_distribution(
        self,
        x_upd_core: NodeVector,
        edge_index_core: Adj,
        edge_attr_core: EdgeVector,
        Z_core: GraphVector,
        Z_block: GraphVector,
        node2graph_core: Optional[LongTensor] = None
        ) -> FloatTensor:
        """
        Input:
            h_upd_core: From core_molecule_embedding    (V_core, Fh_core)
            edge_index_core: From input data            (2, E)
            edge_attr_core: From input data             (E, Fe)
            Z_core: From core_molecule_embedding        (N, Fz_core)
            Z_block: From building_block_embedding      (N, Fz_block)
            node2graph_core: From input data,
        Output:
            P_atom: Probability Distribution of Atoms   (V_core, )
        """
        return self.atom_selection_model(
            x_upd_core, edge_index_core, edge_attr_core,
            Z_core, Z_block, node2graph_core
        )

    def get_atom_logit(
        self,
        x_upd_core: NodeVector,
        edge_index_core: Adj,
        edge_attr_core: EdgeVector,
        Z_core: GraphVector,
        Z_block: GraphVector,
        node2graph_core: Optional[LongTensor] = None,
        ) -> FloatTensor:

        return self.atom_selection_model(
            x_upd_core, edge_index_core, edge_attr_core,
            Z_core, Z_block, node2graph_core, return_logit=True
        )

    def initialize_parameter(self) :
        for param in self.parameters() :
            if param.dim() == 1 :
                continue
            else :
                nn.init.xavier_normal_(param)
    
    def save(self, save_path) :
        torch.save({'model_state_dict': self.state_dict(),
                    'config': self._cfg,
                    'property_information': self.property_information
        }, save_path)
        
    @classmethod
    def load_from_file(cls, checkpoint_path, map_location='cpu') :
        checkpoint = torch.load(checkpoint_path, map_location = map_location)
        return cls.load_from_checkpoint(checkpoint, map_location)

    @classmethod
    def load_from_checkpoint(cls, checkpoint, map_location='cpu') :
        model = cls(checkpoint['config'],checkpoint['property_information'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(map_location)
        return model
