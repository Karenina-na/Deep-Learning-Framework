import torch
from torch import nn


class CA(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction,
                                  kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel,
                             kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel,
                             kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # x: [b, c, h, w]

        # 对高度和宽度分别做全局平均池化
        # [b, c, h, w] -> [b, c, h, 1] -> [b, c, 1, h]
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)

        # [b, c, h, w] -> [b, c, 1, w]
        x_w = self.avg_pool_y(x)

        # [b, c, 1, h] + [b, c, 1, w] -> [b, c // reduction, 1, h+w] 共享卷积
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        # 分割卷积 [b, c // reduction, 1, h+w] -> [b, c // reduction, 1, h] + [b, c // reduction, 1, w]
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        # [b, c // reduction, 1, h] -> [b, c, h, 1]
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # [b, c // reduction, 1, w] -> [b, c, 1, w]
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        # [b, c, h, w] * [b, c, h, 1] * [b, c, 1, w]
        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


if __name__ == '__main__':
    x = torch.randn(64, 32, 128, 256)  # b, c, h, w
    ca_model = CA(channel=32, h=128, w=256)
    y = ca_model(x)
    print(y.shape)
