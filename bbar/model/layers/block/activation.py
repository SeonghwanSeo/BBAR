import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from typing import Optional

# CosineCutoff & cosinecutoff
def cosinecutoff(input: Tensor, cutoff: float) :
    return 0.5 * (torch.cos(math.pi * input / cutoff))

class CosineCutoff(nn.Module) :
    def __init__(self, cutoff: float = 10.0) :
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff
        
    def forward(self, edge_distance) :
        return cosinecutoff(edge_distance, self.cutoff)

# ShiftedSoftPlus & shiftedsoftplus
def shiftedsoftplus(input: Tensor) :
    return F.softplus(input) - math.log(2.0)

class ShiftedSoftplus(nn.Module):
    r"""
    Shited-softplus activated function
    """

    def forward(self, input: Tensor) :
        return shiftsoftplus(input)
