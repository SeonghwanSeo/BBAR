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

class Activation(nn.Module) :
    def __init__(self, activation: Callable) :
        super(Activation, self).__init__()
        self.activation = activation

    def forward(self, input) :
        return self.activation(input)

class Linear(nn.Sequential) :
    def __init__(
        self,
        input_dim: int,
        output_dim: int, 
        activation: Union[Callable, str, None] = None,
        bias: bool = True,
        dropout: float = 0.0
    ) :
        self.input_dim = input_dim
        self.output_dim = output_dim
        if dropout > 0 :
            super(Linear, self).__init__(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, output_dim, bias=bias),
            )
        else :
            super(Linear, self).__init__(
                nn.Linear(input_dim, output_dim, bias=bias),
            )

        if activation is not None :
            if isinstance(activation, str) :
                self.append(ACT_LIST[activation]())
            elif isinstance(activation, nn.Module) :
                self.append(activation)
            else :
                self.append(Activation(activation))
