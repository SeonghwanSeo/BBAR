import torch
import torch.nn as nn
from torch import FloatTensor
from . import block

class BlockSelectionModel(nn.Module) :
    def __init__(
        self,
        core_graph_vector_dim: int, 
        block_graph_vector_dim: int,
        hidden_dim: int, 
        dropout: float = 0.0
    ) :

        super(BlockSelectionModel, self).__init__()
        self.mlp = nn.Sequential(
            block.Linear(
                input_dim = core_graph_vector_dim + block_graph_vector_dim,
                output_dim = hidden_dim,
                activation = 'relu',
                dropout = dropout
            ),
            block.Linear(
                input_dim = hidden_dim,
                output_dim = 1,
                activation = 'sigmoid',
                dropout = dropout
            )
        )

    def forward(self, Z_core: FloatTensor, Z_block: FloatTensor) -> FloatTensor :
        """
        Input :
            Z_core: graph vector of core molecule.  (N, F_z_core)
            Z_block: graph vector of block.         (N, F_z_block)

        Output:
            probability value $\in$ [0, 1]          (N, )
        """
        Z_concat = torch.cat([Z_core, Z_block], dim=-1)         # (N, F_z_c + F_z_b)
        return self.mlp(Z_concat).squeeze(-1)                   # (N, F_z_c + F_z_b) -> (N, F_hid) -> (N, )
