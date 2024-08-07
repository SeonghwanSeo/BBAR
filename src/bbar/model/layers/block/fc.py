import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable

from .activation import ShiftedSoftplus

__all__ = [
    'Linear',
]

ACT_LIST = {
    'relu': nn.ReLU,
    'ReLU': nn.ReLU,
    'silu': nn.SiLU,
    'SiLU': nn.SiLU,
    'tanh': nn.Tanh,
    'Tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'Sigmoid': nn.Sigmoid, 
    'shiftedsoftplus': ShiftedSoftplus,
    'ShiftedSoftplus': ShiftedSoftplus,
}
NORM_LIST = {
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}

class Linear(nn.Sequential) :
    def __init__(
        self,
        input_dim: int,
        output_dim: int, 
        activation: Optional[str] = None,
        norm: Optional[str] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) :
        nonlinear_layer = ACT_LIST[activation]() if activation is not None else nn.Identity()
        norm_layer = NORM_LIST[norm](output_dim) if norm is not None else nn.Identity()
        super(Linear, self).__init__(
            nn.Linear(input_dim, output_dim, bias=bias),
            norm_layer,
            nonlinear_layer,
            nn.Dropout(p=dropout)
        )
