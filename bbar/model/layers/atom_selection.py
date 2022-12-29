import torch
import torch.nn as nn

from typing import Optional
from torch import FloatTensor, LongTensor
from torch_geometric.typing import Adj
from torch_scatter.composite import scatter_softmax

from . import block
from .graph_embedding import GraphEmbeddingModel

class AtomSelectionModel(nn.Module) :
    def __init__(
        self,
        core_edge_input_dim: int,
        core_node_vector_dim: int,    # h_core_upd
        core_graph_vector_dim: int,   # Z_core
        block_graph_vector_dim: int,
        hidden_dim: int = 128,
        n_block: int = 2,
        dropout: float = 0.0
    ) :

        super(AtomSelectionModel, self).__init__()    

        self.graph_embedding = GraphEmbeddingModel(
            node_input_dim = hidden_dim,
            edge_input_dim = core_edge_input_dim,
            global_input_dim = core_graph_vector_dim + block_graph_vector_dim,
            hidden_dim = hidden_dim,
            graph_vector_dim = None,
            n_block = n_block,
            dropout = dropout
        )

        self.mlp = nn.Sequential(
            block.Linear(
                input_dim = core_node_vector_dim,
                output_dim = hidden_dim,
                activation = 'relu',
                dropout = dropout
            ),
            block.Linear(
                input_dim = hidden_dim,
                output_dim = 1,
                activation = None,
            )
        )

    def forward(
        self,
        x_upd_core: FloatTensor,
        edge_index_core: Adj,
        edge_attr_core: FloatTensor,
        Z_core: FloatTensor,
        Z_block: FloatTensor,
        node2graph_core: Optional[LongTensor] = None,
        return_logit: bool = False,
    ) -> FloatTensor:
        """
        Input :
            x_upd_core: updated node feature of core molecule   (V_core, Fh)
            edge_index_core: edge index of core molecule        (2, E_core)
            edge_attr_core: edge attr of core molecule          (E_core, Fe)
            Z_core: latent vector of core molecule              (N, Fz_core)
            Z_block: latent vector of block                     (N, Fz_block)
            node2graph_core: map node to graph of core molecule(optional)
                                                                (V_core,)
        Output:
            P: probability distribution of atoms                (V_core, )
        """

        Z_cat = torch.cat([Z_core, Z_block], dim=-1)                # (N, Fz_core + Fz_block)
        x_upd2, _ = self.graph_embedding(x_upd_core, edge_index_core, edge_attr_core,
                global_x = Z_cat, node2graph = node2graph_core)     # (V, Fh)

        logit = self.mlp(x_upd2).squeeze(-1)                        # (V, Fh + Fv) -> (V, Fh) -> (V, )

        if return_logit :
            return logit
        else :
            if node2graph_core is not None :
                P = scatter_softmax(logit, node2graph_core)
            else :
                P = torch.softmax(logit, dim=-1)
            return P
