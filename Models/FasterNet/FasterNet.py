from collections import OrderedDict
from functools import partial
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from Models.FasterNet.Block.module.ConvBNLayer import ConvBNLayer
from Models.FasterNet.Block.FasterNetBlock import FasterNetBlock


class FasterNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000, last_channels=1280, inner_channels: list = [40, 80, 160, 320],
                 blocks: list = [1, 2, 8, 2], bias=False, act='ReLU', n_div=4, forward='slicing', drop_path=0., ):
        super(FasterNet, self).__init__()
        self.embedding = ConvBNLayer(in_channels, inner_channels[0], kernel_size=4, stride=4, bias=bias)

        self.stage1 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[0], bias=bias, act=act, n_div=n_div, forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[0])]))

        self.merging1 = ConvBNLayer(inner_channels[0], inner_channels[1],
                                    kernel_size=2, stride=2, bias=bias)

        self.stage2 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[1], bias=bias, act=act, n_div=n_div, forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[1])]))

        self.merging2 = ConvBNLayer(inner_channels[1], inner_channels[2],
                                    kernel_size=2, stride=2, bias=bias)

        self.stage3 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[2], bias=bias, act=act, n_div=n_div, forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[2])]))

        self.merging3 = ConvBNLayer(inner_channels[2], inner_channels[3], kernel_size=2, stride=2, bias=bias)

        self.stage4 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[3], bias=bias, act=act, n_div=n_div, forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[3])]))

        self.classifier = nn.Sequential(OrderedDict([
            ('global_average_pooling', nn.AdaptiveAvgPool2d(1)),
            ('conv', nn.Conv2d(inner_channels[-1], last_channels, kernel_size=1, bias=False)),
            ('act', getattr(nn, act)()),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(last_channels, out_channels, bias=True))
        ]))
        self.feature_channels = inner_channels

    def fuse_bn_tensor(self):
        for m in self.modules():
            if isinstance(m, ConvBNLayer):
                m._fuse_bn_tensor()

    def forward_feature(self, x: Tensor) -> List[Tensor]:
        x1 = self.stage1(self.embedding(x))
        x2 = self.stage2(self.merging1(x1))
        x3 = self.stage3(self.merging2(x2))
        x4 = self.stage4(self.merging3(x3))
        return [x1, x2, x3, x4]

    def forward(self, x: Tensor) -> Tensor:
        _, _, _, x = self.forward_feature(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = FasterNet(inner_channels=[40, 80, 160, 320], out_channels=200,
                      blocks=[1, 2, 8, 2], act='GELU', drop_path=0.)
    print(model(x).shape)
