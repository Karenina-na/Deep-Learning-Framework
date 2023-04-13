import torch.nn as nn


# 定义判别器
# 输入为1*28*28的图像，输出为0~1的概率，二分类概率值
# BCELoss计算交叉熵
# 判别器中一般推荐使用LeakyReLU激活函数
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),  # 激活函数为LeakyReLU，x>0时，y=x，x<0时，y=0.2x
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 输出范围为[0,1]，激活函数为sigmoid
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        output = self.main(x)
        return output