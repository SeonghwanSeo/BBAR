import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

from typing import Optional, Tuple, Union

from . import fc

# Revised Version of GINEConv.
class GINEConv(MessagePassing) :
    def __init__(
        self,
        node_dim: Union[int, Tuple[int]],
        edge_dim: int,
        activation: Optional[str] = None,
        **kwargs,
    ) :
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)

        if isinstance(node_dim, int) :
            src_node_dim, dst_node_dim = node_dim, node_dim
        else :  # (int, int)
            src_node_dim, dst_node_dim = node_dim

        self.edge_layer = fc.Linear(edge_dim, src_node_dim, activation)

        self.message_layer = fc.Linear(src_node_dim, dst_node_dim, activation)
        self.eps = torch.nn.Parameter(torch.Tensor([0.1]))

        self.out_layer = fc.Linear(dst_node_dim, dst_node_dim, activation)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Tensor) -> Tensor :
        """
        x: node feature             [(V_src, Fh_src), (V_dst, Fh_dst)]
        edge_index: edge index      (2, E)
        edge_attr: edge feature     (E, Fe)
        """
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_src, x_dst = x

        x_dst_upd = self.propagate(edge_index, x = x, edge_attr = edge_attr)
        
        if x_dst is not None :
            x_dst_upd = x_dst_upd + (1 + self.eps) * x_dst

        return self.out_layer(x_dst_upd)

    def message(self, x_j, edge_attr) :
        # x_i: dst, x_j: src
        edge_attr = self.edge_layer(edge_attr)
        return self.message_layer((x_j + edge_attr).relu())    # (E, Fh_dst)
