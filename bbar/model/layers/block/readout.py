import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch_scatter import scatter_sum, scatter_mean, scatter_max
from typing import Optional

from . import fc

class Readout(nn.Module) :
    """
    Input
    nodes : n_node, node_dim
    global_x: n_graph, global_dim 

    Output(Graph Vector)
    retval : n_graph, output_dim
    """
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        output_dim: int,
        global_input_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Optional[str] = None
    ) :

        super(Readout, self).__init__()
        global_input_dim = global_input_dim or 0

        self.linear1 = fc.Linear(node_dim, hidden_dim, bias = False, dropout = dropout)

        self.linear2 = fc.Linear(node_dim, hidden_dim, activation = 'Sigmoid', dropout = dropout)
        self.linear3 = fc.Linear(hidden_dim*3 + global_input_dim, output_dim, activation, dropout = dropout)

    def forward(self, x: FloatTensor, node2graph: Optional[LongTensor] = None,
                            global_x: Optional[FloatTensor] = None) -> FloatTensor:
        """
        x: [V, Fh]
        node2graph: optional, [V, ]
        global_x: optional, [N, Fc]
        """
        x = self.linear1(x) * self.linear2(x)               # Similar to SiLU   SiLU(x) = x * sigmoid(x)
        if node2graph is not None :
            Z1 = scatter_sum(x, node2graph, dim=0)          # V, Fh -> N, Fz
            Z2 = scatter_mean(x, node2graph, dim=0)         # V, Fh -> N, Fz
            Z3 = scatter_max(x, node2graph, dim=0)[1]       # V, Fh -> N, Fz
        else :  # when N = 1
            Z1 = x.sum(dim=0, keepdim = True)               # V, Fh -> 1, Fz       
            Z2 = x.mean(dim=0, keepdim = True)              # V, Fh -> 1, Fz       
            Z3 = x.max(dim=0, keepdim = True)[1]            # V, Fh -> 1, Fz
        if global_x is not None :
            Z = torch.cat([Z1, Z2, Z3, global_x], dim=-1)   # N, 3*Fh + F_cond
        else :
            Z = torch.cat([Z1, Z2, Z3], dim=-1)             # N, 3*Fh
        return self.linear3(Z)                              # N, 3*Fh + F_cond -> N, Fh
