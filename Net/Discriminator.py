import torch
import torch.nn as nn

# 输入通道
in_channels = 2


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # 第一个滑动卷积层，不使用BN，LRelu激活函数
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 第二个滑动卷积层，包含BN，LRelu激活函数
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 第三个滑动卷积层，包含BN，LRelu激活函数
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 第四个滑动卷积层，包含BN，LRelu激活函数
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=1, ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 调整维度
            torch.nn.AdaptiveAvgPool1d(1)
        )
        # 全连接层+Sigmoid激活函数
        self.Linear = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = x.view(-1, 128)
        # print(x.shape)
        x = self.Linear(x)
        return x


if __name__ == "__main__":
    x = torch.randn(125, 600, 2)
    D = Discriminator()
    y = D(x.permute(0, 2, 1))
    print(y.shape)
