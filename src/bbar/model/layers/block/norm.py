import torch
import torch.nn as nn

import torch_geometric.nn as pygnn

class GraphNorm(pygnn.LayerNorm):
    def __init__(self, in_channels: int, eps: float = 1e-5, affine: bool = True) :
        super(GraphNorm, self).__init__(in_channels, eps, affine, mode='graph')
