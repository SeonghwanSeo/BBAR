import torch
import torch.nn as nn

from typing import Tuple, Dict, OrderedDict, Union
from torch import FloatTensor, LongTensor
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from bbar.utils.typing import NodeVector, EdgeVector, GraphVector, GlobalVector

from .layers import GraphEmbeddingModel, ConditionEmbeddingModel, \
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
        self.condition_dim = len(property_information)

        # Graph Embedding
        self.core_graph_embedding_model = GraphEmbeddingModel(NUM_ATOM_FEATURES, NUM_BOND_FEATURES,
                                            0, **cfg.GraphEmbeddingModel_Core)

        self.block_graph_embedding_model = GraphEmbeddingModel(NUM_ATOM_FEATURES, NUM_BOND_FEATURES,
                                            NUM_BLOCK_FEATURES, **cfg.GraphEmbeddingModel_Block)

        # Condition Embedding
        self.condition_embedding_model = ConditionEmbeddingModel(condition_dim = self.condition_dim, \
                                            **cfg.ConditionEmbeddingModel)

        # Terminate Prediction
        self.termination_prediction_model = TerminationPredictionModel(**cfg.TerminationPredictionModel)

        # Building Block Section
        self.block_selection_model = BlockSelectionModel(**cfg.BlockSelectionModel)

        # Atom Selection
        self.atom_selection_model = AtomSelectionModel(NUM_ATOM_FEATURES, NUM_BOND_FEATURES,
                                            **cfg.AtomSelectionModel)

    def _standardize_condition(
        self,
        condition: Dict[str, Union[float, FloatTensor]],
        device: torch.device = 'cpu',
    ) -> GlobalVector :

        assert condition.keys() == self.property_keys, \
            f"Input Keys is not valid\n" \
            f"\tInput:      {set(condition.keys())}\n" \
            f"\tRequired:   {set(self.property_keys)}"

        condition = [(condition[key] - mean) / std
                                for key, (mean, std) in self.property_information.items()]
        if isinstance(condition[0], float) :
            condition = torch.FloatTensor([condition], device=device)     # (1, Fc)
        else :
            condition = torch.stack(condition, dim=-1)                    # (N, Fc)

        return condition
        
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
    
    def condition_embedding(
        self,
        x_upd_core: NodeVector,
        Z_core: GraphVector,
        condition: Dict[str, Union[float, FloatTensor]],
        node2graph_core: LongTensor,
        condition_noise: float = 0.0,
    ) -> Tuple[NodeVector, GraphVector]:
        """
        Input:
            x_upd_core: Updated Node Vector             (V_core, Fh_core)
            Z_core: Graph Vector                        (N, Fz_core)
            condition: Condition                        {property_key: property_value}

        Output:
            x_upd_core: Updated Node Vector             (V_core, Fh_core)
            Z_core: Graph Vector                        (N, Fz_core)
        """
        condition_vector = self._standardize_condition(condition)   # (N, F_condition)
        if condition_noise > 0 :
            noise = torch.randn_like(condition) * condition_noise
            condition_vector += noise
        return self.condition_embedding_model(x_upd_core, Z_core, condition_vector, node2graph_core)

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
        batch_core: Union[PyGData, PyGBatch],
        x_upd_core: NodeVector,
        Z_core: GraphVector,
        Z_block: GraphVector,
        ) -> FloatTensor:
        """
        Input:
            batch_core: PyGData or PyGBatch. (Transform by CoreGraphTransform) 
            h_upd_core: From core_molecule_embedding    (V_core, Fh_core)
            Z_core: From core_molecule_embedding        (N, Fz_core)
            Z_block: From building_block_embedding      (N, Fz_block)
        Output:
            P_atom: Probability Distribution of Atoms   (V_core, )
        """
        x_inp_core, edge_index_core, edge_attr_core = \
                batch_core.x, batch_core.edge_index, batch_core.edge_attr
        if isinstance(batch_core, PyGBatch) :
            node2graph_core = batch_core.batch
        else :
            node2graph_core = None

        return self.atom_selection_model(
            x_inp_core, edge_index_core, edge_attr_core, x_upd_core, 
            Z_core, Z_block, node2graph_core
        )

    def get_atom_logit(
        self,
        batch_core: Union[PyGData, PyGBatch],
        x_upd_core: NodeVector,
        Z_core: GraphVector,
        Z_block: GraphVector,
        ) -> FloatTensor:
        x_inp_core, edge_index_core, edge_attr_core = \
                batch_core.x, batch_core.edge_index, batch_core.edge_attr
        if isinstance(batch_core, PyGBatch) :
            node2graph_core = batch_core.batch
        else :
            node2graph_core = None

        return self.atom_selection_model(
            x_inp_core, edge_index_core, edge_attr_core, x_upd_core, 
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
