import torch.nn as nn
import torch.nn.functional as F
from Models.MobileNet.Block.SEBlock_2d import SE


def Hswish(x, inplace=True):
    return x * F.relu6(x + 3., inplace=inplace) / 6.


def Hsigmoid(x, inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_channels, stride, se='True', nl='HS'):
        super(Bottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride = stride
        if se:
            self.se = SE(exp_channels, function=Hsigmoid)
        else:
            self.se = None
        self.conv1 = nn.Conv2d(in_channels, exp_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2 = nn.Conv2d(exp_channels, exp_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=exp_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_channels)
        self.conv3 = nn.Conv2d(exp_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # 先初始化一个空序列，之后改造其成为残差链接
        self.shortcut = nn.Sequential()
        # 只有步长为1且输入输出通道不相同时才采用跳跃连接(想一下跳跃链接的过程，输入输出通道相同这个跳跃连接就没意义了)
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 下面的操作卷积不改变尺寸，仅匹配通道数
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out = self.bn2(self.conv2(out))
            out = self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
