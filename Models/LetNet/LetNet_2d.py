import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 调用父类的构造函数
        self.conv1 = nn.Conv2d(3, 16, 5)  # 卷积层：深度为3的特征矩阵，使用16个卷积核，卷积核的尺寸为5*5
        self.pool1 = nn.MaxPool2d(2, 2)  # 下采样层：池化核为2*2，布局为2的池化层
        self.conv2 = nn.Conv2d(16, 32, 5)  # 卷积层
        self.pool2 = nn.MaxPool2d(2, 2)  # 下采样层
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # 全连接层：输入是一个一维向量，需要将特征矩阵展平
        self.fc2 = nn.Linear(120, 84)  # 全连接层
        self.fc3 = nn.Linear(84, 10)  # 全连接层

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        """ 经卷积后的矩阵尺寸大小计算公式为：N= (W - F + 2P)/S+1 
            W为输入图片大小W*W
            F为卷积核的尺寸
            S为步长Stride=1
            P为Padding=0
        """
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10) ;池化层只改变矩阵的高和宽，不会影响深度(28-2)/2+1
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x
