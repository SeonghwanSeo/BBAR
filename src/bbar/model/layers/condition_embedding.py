import torch
import torch.nn as nn
from torch import FloatTensor
from . import block

from typing import Tuple, Optional

class ConditionEmbeddingModel(nn.Module) :
    def __init__(
        self,
        core_node_vector_dim: int,
        core_graph_vector_dim: int,
        condition_dim: int,
        dropout: float = 0.0
    ) :

        super(ConditionEmbeddingModel, self).__init__()
        self.node_mlp = nn.Sequential(
            block.Linear(
                input_dim = core_node_vector_dim + condition_dim,
                output_dim = core_node_vector_dim,
                activation = 'SiLU',
                dropout = dropout
            ),
            block.Linear(
                input_dim = core_node_vector_dim,
                output_dim = core_node_vector_dim,
                activation = 'SiLU',
                dropout = dropout
            )
        )
        self.graph_mlp = nn.Sequential(
            block.Linear(
                input_dim = core_graph_vector_dim + condition_dim,
                output_dim = core_graph_vector_dim,
                activation = 'SiLU',
                dropout = dropout
            ),
            block.Linear(
                input_dim = core_graph_vector_dim,
                output_dim = core_graph_vector_dim,
                activation = 'SiLU',
                dropout = dropout
            )
        )

    def forward(self, h_core: FloatTensor, Z_core: FloatTensor, condition: FloatTensor, \
                node2graph: Optional[FloatTensor]) -> Tuple[FloatTensor, FloatTensor]:
        """
        Input :
            h_core: node vector of core molecule.   (V, F_h_core)
            Z_core: graph vector of core molecule.  (N, F_z_core)
            condition: condition                    (N, F_condition)

        Output:
            h_core: node vector of core molecule.   (V, F_h_core)
            Z_core: graph vector of core molecule.  (N, F_z_core)
        """
        if node2graph is not None :
            h_condition = condition[node2graph]                 # (N, F_condition) -> (V, F_condition)
        else :
            h_condition = condition.repeat(h_core.size(0), 1)   # (1, F_condition) -> (V, F_condition)
        Z_condition = condition                             # (N, F_condition)
        h_core = torch.cat([h_core, h_condition], dim=-1)       # (V, F_h_core + F_condition)
        Z_core = torch.cat([Z_core, Z_condition], dim=-1)       # (V, F_z_core + F_condition)
        return self.node_mlp(h_core), self.graph_mlp(Z_core)    # (V, F_h_core), (N, F_z_core)
