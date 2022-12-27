import torch
import torch.nn as nn
from torch import FloatTensor
from . import block

class PropertyPredictionModel(nn.Module) :
    def __init__(
        self,
        core_graph_vector_dim: int,
        property_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0
    ) :
        super(PropertyPredictionModel, self).__init__()    
        self.mlp = nn.Sequential(
            block.Linear(
                input_dim = core_graph_vector_dim,
                output_dim = hidden_dim,
                activation = 'relu',
                dropout = dropout
            ),
            block.Linear(
                input_dim = hidden_dim,
                output_dim = hidden_dim,
                activation = 'relu',
                dropout = dropout
            ),
            block.Linear(
                input_dim = hidden_dim,
                output_dim = property_dim,
                activation = None,
                dropout = dropout
            )
        )

    def forward(self, Z_core: FloatTensor) -> FloatTensor:
        """
        Input :
            Z_core: graph vector of core molecule.  (N, F_z_core)

        Output:
            predicted property value
        """
        y = self.mlp(Z_core)            # (N, Fz) -> (N, Fh) -> (N, Fc)
        return y
