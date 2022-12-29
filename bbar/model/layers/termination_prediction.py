import torch
import torch.nn as nn
from torch import FloatTensor
from . import block

class TerminationPredictionModel(nn.Module) :
    def __init__(
        self,
        core_graph_vector_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0
    ) :

        super(TerminationPredictionModel, self).__init__()    
        self.mlp = nn.Sequential(
            block.Linear(
                input_dim = core_graph_vector_dim,
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, Z_core: FloatTensor, return_logit = False) -> FloatTensor:
        """
        Input :
            Z_core: graph vector of core molecule.  (N, F_z_core)

        Output:
            probability value $\in$ [0, 1]          (N, )
        """
        logit = self.mlp(Z_core).squeeze(-1)        # (N, F_z_c) -> (N, F_h) -> (N, )
        if return_logit is False :
            return self.sigmoid(logit)
        else :
            return logit
