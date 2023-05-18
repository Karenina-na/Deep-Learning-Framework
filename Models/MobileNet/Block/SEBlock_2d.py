import torch
from torch import nn


class SE(nn.Module):
    # ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4, function=nn.Sigmoid):
        super().__init__()

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)

        # relu激活，可自行换别的激活函数
        self.relu = nn.ReLU()

        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = function

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)

        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)

        x = self.relu(x)

        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)

        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs


if __name__ == "__main__":
    inputs = torch.randn([2, 64, 32, 32])
    se = SE(64)
    outputs = se(inputs)
    print(outputs.shape)
