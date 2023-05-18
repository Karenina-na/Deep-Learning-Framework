import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.MobileNet.Block.SEBlock_2d import SE
from Models.MobileNet.Block.InvertedResidualConvSE_2d import Hsigmoid,Hswish, Bottleneck


class MobileNetV3_large(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg = [
        (16, 3, 16, 1, False, 'RE'),
        (24, 3, 64, 2, False, 'RE'),
        (24, 3, 72, 1, False, 'RE'),
        (40, 5, 72, 2, True, 'RE'),
        (40, 5, 120, 1, True, 'RE'),
        (40, 5, 120, 1, True, 'RE'),
        (80, 3, 240, 2, False, 'HS'),
        (80, 3, 200, 1, False, 'HS'),
        (80, 3, 184, 1, False, 'HS'),
        (80, 3, 184, 1, False, 'HS'),
        (112, 3, 480, 1, True, 'HS'),
        (112, 3, 672, 1, True, 'HS'),
        (160, 5, 672, 2, True, 'HS'),
        (160, 5, 960, 1, True, 'HS'),
        (160, 5, 960, 1, True, 'HS')
    ]

    def __init__(self, num_classes=17):
        super(MobileNetV3_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2 = nn.Conv2d(160, 960, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3 = nn.Conv2d(960, 1280, 1, 1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(1280, num_classes, 1, stride=1, padding=0, bias=True)

    def _make_layers(self, in_channels):
        layers = []
        for out_channels, kernel_size, exp_channels, stride, se, nl in self.cfg:
            layers.append(
                Bottleneck(in_channels, out_channels, kernel_size, exp_channels, stride, se, nl)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = Hswish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = Hswish(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a, b = out.size(0), out.size(1)
        out = out.view(a, b)
        return out


class MobileNetV3_small(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg = [
        (16, 3, 16, 2, True, 'RE'),
        (24, 3, 72, 2, False, 'RE'),
        (24, 3, 88, 1, False, 'RE'),
        (40, 5, 96, 2, True, 'HS'),
        (40, 5, 240, 1, True, 'HS'),
        (40, 5, 240, 1, True, 'HS'),
        (48, 5, 120, 1, True, 'HS'),
        (48, 5, 144, 1, True, 'HS'),
        (96, 5, 288, 2, True, 'HS'),
        (96, 5, 576, 1, True, 'HS'),
        (96, 5, 576, 1, True, 'HS')
    ]

    def __init__(self, num_classes=17):
        super(MobileNetV3_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2 = nn.Conv2d(96, 576, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3 = nn.Conv2d(576, 1280, 1, 1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(1280, num_classes, 1, stride=1, padding=0, bias=True)

    def _make_layers(self, in_channels):
        layers = []
        for out_channels, kernel_size, exp_channels, stride, se, nl in self.cfg:
            layers.append(
                Bottleneck(in_channels, out_channels, kernel_size, exp_channels, stride, se, nl)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = Hswish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.bn2(self.conv2(out))
        se = SE(out.size(1), function=Hsigmoid)
        out = Hswish(se(out))
        out = F.avg_pool2d(out, 7)
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a, b = out.size(0), out.size(1)
        out = out.view(a, b)
        return out


if __name__ == "__main__":
    net_small = MobileNetV3_small(2)
    net_large = MobileNetV3_large(2)
    x = torch.randn(2, 3, 224, 224)
    y1 = net_small(x)
    y2 = net_large(x)
    print(y1.shape)
    print(y2.shape)
