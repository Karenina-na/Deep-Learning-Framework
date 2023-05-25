import torch
import torch.nn as nn
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple
import numpy as np
from Models.DenseNet.block.DenBlock import _DenseBlock as DenseBlock
from Models.DenseNet.block.Transition import _TransitionLayer as TransitionLayer


class DenseNet(nn.Module):
    def __init__(self, num_init_features=64, growth_rate=32, blocks=(6, 12, 24, 16), bn_size=4, drop_rate=0,
                 num_classes=1000):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        self.layer1 = DenseBlock(num_layers=blocks[0], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transtion1 = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer2 = DenseBlock(num_layers=blocks[1], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transtion2 = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer3 = DenseBlock(num_layers=blocks[2], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transtion3 = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer4 = DenseBlock(num_layers=blocks[3], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x -> [batch_size, 3, 224, 224]

        x = self.features(x)  # x -> [batch_size, 64, 56, 56]

        x = self.layer1(x)  # x -> [batch_size, 256, 56, 56] DenseBlock

        x = self.transtion1(x)  # x -> [batch_size, 128, 28, 28] TransitionLayer

        x = self.layer2(x)  # x -> [batch_size, 512, 28, 28] DenseBlock

        x = self.transtion2(x)  # x -> [batch_size, 256, 14, 14] TransitionLayer

        x = self.layer3(x)  # x -> [batch_size, 1024, 14, 14] DenseBlock

        x = self.transtion3(x)  # x -> [batch_size, 512, 7, 7] TransitionLayer

        x = self.layer4(x)  # x -> [batch_size, 2048, 7, 7] DenseBlock

        x = self.avgpool(x)  # x -> [batch_size, 2048, 1, 1]
        x = torch.flatten(x, start_dim=1)  # x -> [batch_size, 2048]
        x = self.fc(x)  # x -> [batch_size, classes]

        return x


if __name__ == '__main__':
    model = DenseNet(blocks=(6, 12, 64, 48), num_classes=5)
    x = torch.randn(25, 3, 224, 224)
    y = model(x)
    print(y.shape)
