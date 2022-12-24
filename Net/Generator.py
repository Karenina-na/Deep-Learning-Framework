import torch
import torch.nn as nn

# 输入通道
in_channels = 2


# 生成器
class Generator(nn.Module):
    # 噪声维度
    def __init__(self, input_size):
        super(Generator, self).__init__()

        # 第一层：把输入线性变换成256x4x4的矩阵，并在这个基础上做反卷机操作
        self.Linear = nn.Sequential(nn.Linear(input_size, 256 * 75))

        self.deconv = nn.Sequential(

            # 第二层：bn+relu
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0, ),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            # 第三层：bn+relu
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, ),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            # 第四层:不使用BN，使用tanh激活函数
            nn.ConvTranspose1d(in_channels=64, out_channels=in_channels, kernel_size=4, stride=2, padding=2, ),
            nn.Tanh()
        )

    def forward(self, x):
        # 把随机噪声经过线性变换
        x = self.Linear(x)
        # print(x.shape)
        x = x.view([-1, 256, 75])
        # print(x.shape)
        # 反卷积
        x = self.deconv(x)
        # 调转维度
        return x.permute(0, 2, 1)


if __name__ == "__main__":
    D = Generator(100)
    ran = torch.randn(100)
    y = D(ran)
    print(y.shape)
