import torch.nn as nn
import torch


class Generator(nn.Module):
    """反滑动卷积生成器"""

    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        layers = []

        # 第一层：把输入线性变换成256x4x4的矩阵，并在这个基础上做反卷机操作
        self.linear = nn.Linear(self.z_dim, 16 * 16 * 512)
        # 第二层：bn+relu
        layers.append(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=0))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        # 第三层：bn+relu
        layers.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        # 第四层：bn+relu
        layers.append(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        # 第六层:不使用BN，使用tanh激活函数
        layers.append(nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=2))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        # 把随机噪声经过线性变换，resize成256x4x4的大小
        x = self.linear(z)
        x = x.view([x.size(0), 512, 16, 16])
        # 生成图片
        x = self.model(x)
        return x


if __name__ == "__main__":
    data = torch.randn(1, 100)
    model = Generator(100)
    print(model(data).shape)



