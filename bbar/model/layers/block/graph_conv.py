import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

from typing import Optional, Tuple, Union

from . import fc

class ResidualBlock(nn.Module) :
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        activation: str = 'SiLU',
        norm: bool = True,
        ) :
        super(ResidualBlock, self).__init__()
        self.conv1 = GINEConv(node_dim = node_dim, edge_dim = edge_dim,
                                                            activation = activation)
        self.graph_norm1 = pyg_nn.LayerNorm(in_channels=node_dim, mode='graph') if norm is True else None

        self.conv2 = GINEConv(node_dim = node_dim, edge_dim = edge_dim,
                                                            activation = activation)
        self.graph_norm2 = pyg_nn.LayerNorm(in_channels=node_dim, mode='graph') if norm is True else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, node2graph: OptTensor) :
        identity = x

        out = self.conv1(x, edge_index, edge_attr = edge_attr)
        if self.graph_norm1 is not None :
            out = self.graph_norm1(out, node2graph)
        out = self.relu(out)

        out = self.conv2(x, edge_index, edge_attr = edge_attr)
        if self.graph_norm2 is not None :
            out = self.graph_norm2(out, node2graph)

        out += identity
        out = self.relu(out)
        
        return out

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

        return x_dst_upd 

    def message(self, x_j, edge_attr) :
        # x_i: dst, x_j: src
        edge_attr = self.edge_layer(edge_attr)
        return self.message_layer((x_j + edge_attr).relu())    # (E, Fh_dst)