import torch
import torch.nn as nn
from torch import Tensor

from Models.FasterNet.Block.module.ConvBNLayer import ConvBNLayer
from Models.FasterNet.Block.module.PConv import PConv
from Models.FasterNet.Block.module.DropPath import DropPath


class FasterNetBlock(nn.Module):
    def __init__(self, in_channels: int, inner_channels: int = None, kernel_size: int = 3, bias=False,
                 act: str = 'ReLU', n_div: int = 4, forward: str = 'split_cat', drop_path: float = 0., ):
        super(FasterNetBlock, self).__init__()
        inner_channels = inner_channels or in_channels * 2
        self.conv1 = PConv(in_channels, kernel_size, n_div, forward)
        self.conv2 = ConvBNLayer(in_channels, inner_channels, bias=bias, act=act)
        self.conv3 = nn.Conv2d(inner_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        return x + self.drop_path(y)
