import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, List, Tuple
import numpy as np
from Models.DenseNet.block.DenseLayer import _DenseLayer


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
