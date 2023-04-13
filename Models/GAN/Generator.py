import torch.nn as nn


# 定义生成器
# 输入为100正态分布随机数（噪声），输出为1*28*28的图像
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),  # 输出范围为[-1,1]，激活函数为tanh
        )

    def forward(self, x):  # 前向传播
        output = self.main(x)
        output = output.view(-1, 28, 28)  # 将输出转换为1*28*28的图像
        return output
