import torch
import torch.nn as nn
import torchvision


def BottleneckV1(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                  groups=in_channels),  # deepwise
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),  # pointwise
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )
