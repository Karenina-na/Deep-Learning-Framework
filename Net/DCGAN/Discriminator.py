import torch.nn as nn
import torch


class Discriminator(nn.Module):
    """滑动卷积判别器"""

    def __init__(self):
        super(Discriminator, self).__init__()

        # 定义卷积层
        conv = []
        # 第一个滑动卷积层，不使用BN，LRelu激活函数
        conv.append(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1))
        conv.append(nn.LeakyReLU(0.2, inplace=True))
        # 第二个滑动卷积层，包含BN，LRelu激活函数
        conv.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1))
        conv.append(nn.BatchNorm2d(32))
        conv.append(nn.LeakyReLU(0.2, inplace=True))
        # 第三个滑动卷积层，包含BN，LRelu激活函数
        conv.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        conv.append(nn.BatchNorm2d(64))
        conv.append(nn.LeakyReLU(0.2, inplace=True))
        # 第四个滑动卷积层，包含BN，LRelu激活函数
        conv.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
        conv.append(nn.BatchNorm2d(128))
        conv.append(nn.LeakyReLU(0.2, inplace=True))
        # 卷积层
        self.conv = nn.Sequential(*conv)

        # 全连接层+Sigmoid激活函数
        self.linear = nn.Sequential(nn.Linear(in_features=32768, out_features=1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32768)
        validity = self.linear(x)
        return validity


if __name__ == "__main__":
    data = torch.randn(64, 3, 256, 256)
    model = Discriminator()
    print(model(data).shape)
