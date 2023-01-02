import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# conv + bn + relu
class BasicConv1d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# stem v2
class Stem(nn.Module):
    def __init__(self, in_channel):
        super(Stem, self).__init__()
        self.conv1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=3, stride=2, padding=0)
        self.conv2 = BasicConv1d(in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = BasicConv1d(in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1)

        self.branch1_1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.branch1_2 = BasicConv1d(in_channel=64, out_channel=96, kernel_size=3, stride=2, padding=0)

        self.branch2_1_1 = BasicConv1d(in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0)
        self.branch2_1_2 = BasicConv1d(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0)

        self.branch2_2_1 = BasicConv1d(in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0)
        self.branch2_2_2 = BasicConv1d(in_channel=64, out_channel=64, kernel_size=7, stride=1, padding=3)
        self.branch2_2_3 = BasicConv1d(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0)

        self.branch3_1 = BasicConv1d(in_channel=192, out_channel=192, kernel_size=3, stride=2, padding=0)
        self.branch3_2 = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4_1 = self.branch1_1(out3)
        out4_2 = self.branch1_2(out3)
        out4 = torch.cat((out4_1, out4_2), dim=1)

        out5_1 = self.branch2_1_2(self.branch2_1_1(out4))
        out5_2 = self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out4)))
        out5 = torch.cat((out5_1, out5_2), dim=1)

        out6_1 = self.branch3_1(out5)
        out6_2 = self.branch3_2(out5)
        out = torch.cat((out6_1, out6_2), dim=1)
        return out


# Inception-ResNet-A  out 384
class InceptionResNetA(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionResNetA, self).__init__()
        self.branch1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1)

        self.branch3_1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = BasicConv1d(in_channel=32, out_channel=48, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = BasicConv1d(in_channel=48, out_channel=64, kernel_size=3, stride=1, padding=1)

        self.linear = BasicConv1d(in_channel=128, out_channel=384, kernel_size=1, stride=1, padding=0)
        self.out_channel = 384
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        output1 = self.branch1(x)

        output2 = self.branch2_2(self.branch2_1(x))

        output3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))

        out = torch.cat((output1, output2, output3), dim=1)
        out = self.linear(out)

        x = self.shortcut(x)
        out = x + self.scale * out
        return out


# Reduction-A
class ReductionA(nn.Module):
    def __init__(self, in_channel):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.branch2 = BasicConv1d(in_channel=in_channel, out_channel=384, kernel_size=3, stride=2, padding=0)

        self.branch3_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = BasicConv1d(in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = BasicConv1d(in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        out1 = self.branch1(x)

        out2 = self.branch2(x)

        out3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))

        # print(out1.shape, out2.shape, out3.shape)
        return torch.cat((out1, out2, out3), dim=1)


# Inception-ResNet-B
class InceptionResNetB(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionResNetB, self).__init__()
        self.branch1 = BasicConv1d(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=128, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=128, out_channel=192, kernel_size=7, stride=1, padding=3)

        self.linear = BasicConv1d(in_channel=384, out_channel=1152, kernel_size=1, stride=1, padding=0)
        self.out_channel = 1152
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        output1 = self.branch1(x)

        output2 = self.branch2_2(self.branch2_1(x))
        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)

        x = self.shortcut(x)
        out = x + out * self.scale
        out = F.relu(out)
        return out


# Reduction-B
class ReductionB(nn.Module):
    def __init__(self, in_channel):
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0)

        self.branch3_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = BasicConv1d(in_channel=256, out_channel=288, kernel_size=3, stride=2, padding=0)

        self.branch4_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch4_2 = BasicConv1d(in_channel=256, out_channel=288, kernel_size=3, stride=1, padding=1)
        self.branch4_3 = BasicConv1d(in_channel=288, out_channel=320, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        out1 = self.branch1(x)

        out2 = self.branch2_2(self.branch2_1(x))

        out3 = self.branch3_2(self.branch3_1(x))

        out4 = self.branch4_3(self.branch4_2(self.branch4_1(x)))

        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        return torch.cat((out1, out2, out3, out4), dim=1)


# Inception-ResNet-C
class InceptionResNetC(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionResNetC, self).__init__()
        self.branch1 = BasicConv1d(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=192, out_channel=256, kernel_size=3, stride=1, padding=1)

        self.linear = BasicConv1d(in_channel=448, out_channel=2144, kernel_size=1, stride=1, padding=0)
        self.out_channel = 2144
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        output1 = self.branch1(x)

        output2 = self.branch2_2(self.branch2_1(x))

        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)

        x = self.shortcut(x)
        out = x + out * self.scale
        out = F.relu(out)
        return out


# Inception_ResNet_v2
class InceptionResNetV2(nn.Module):
    def __init__(self):
        super(InceptionResNetV2, self).__init__()
        # stem
        blocks = [Stem(n_channels)]
        # Inception-ResNet-A
        for _ in range(inception_A_block_num):
            blocks.append(InceptionResNetA(384))
        # Reduction-A
        blocks.append(ReductionA(384))
        # Inception-ResNet-B
        for _ in range(inception_B_block_num):
            blocks.append(InceptionResNetB(1152))
        # Reduction-B
        blocks.append(ReductionB(1152))
        # Inception-ResNet-C
        for _ in range(inception_C_block_num):
            blocks.append(InceptionResNetC(2144))

        self.map = nn.Sequential(*blocks)
        self.pool = nn.AvgPool1d(kernel_size=8)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2144*2, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.map(x)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.linear(out.view(-1, 2144*2))
        out = self.softmax(out)
        return out


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1].cpu().numpy()
    y_label = torch.max(labels, 1)[1].data.cpu().numpy()
    rights = (pred == y_label).sum()
    return rights, len(labels)


n_channels = 2
n_classes = 7
inception_A_block_num = 5
inception_B_block_num = 10
inception_C_block_num = 5

if __name__ == '__main__':
    a = torch.randn(64, 2, 600)
    net = InceptionResNetV2()
    print(net(a).shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# conv + bn + relu
class BasicConv1d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# stem v2
class Stem(nn.Module):
    def __init__(self, in_channel):
        super(Stem, self).__init__()
        self.conv1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=3, stride=2, padding=0)
        self.conv2 = BasicConv1d(in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = BasicConv1d(in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1)

        self.branch1_1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.branch1_2 = BasicConv1d(in_channel=64, out_channel=96, kernel_size=3, stride=2, padding=0)

        self.branch2_1_1 = BasicConv1d(in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0)
        self.branch2_1_2 = BasicConv1d(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0)

        self.branch2_2_1 = BasicConv1d(in_channel=160, out_channel=64, kernel_size=1, stride=1, padding=0)
        self.branch2_2_2 = BasicConv1d(in_channel=64, out_channel=64, kernel_size=7, stride=1, padding=3)
        self.branch2_2_3 = BasicConv1d(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0)

        self.branch3_1 = BasicConv1d(in_channel=192, out_channel=192, kernel_size=3, stride=2, padding=0)
        self.branch3_2 = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4_1 = self.branch1_1(out3)
        out4_2 = self.branch1_2(out3)
        out4 = torch.cat((out4_1, out4_2), dim=1)

        out5_1 = self.branch2_1_2(self.branch2_1_1(out4))
        out5_2 = self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out4)))
        out5 = torch.cat((out5_1, out5_2), dim=1)

        out6_1 = self.branch3_1(out5)
        out6_2 = self.branch3_2(out5)
        out = torch.cat((out6_1, out6_2), dim=1)
        return out


# Inception-ResNet-A  out 384
class InceptionResNetA(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionResNetA, self).__init__()
        self.branch1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1)

        self.branch3_1 = BasicConv1d(in_channel=in_channel, out_channel=32, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = BasicConv1d(in_channel=32, out_channel=48, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = BasicConv1d(in_channel=48, out_channel=64, kernel_size=3, stride=1, padding=1)

        self.linear = BasicConv1d(in_channel=128, out_channel=384, kernel_size=1, stride=1, padding=0)
        self.out_channel = 384
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        output1 = self.branch1(x)

        output2 = self.branch2_2(self.branch2_1(x))

        output3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))

        out = torch.cat((output1, output2, output3), dim=1)
        out = self.linear(out)

        x = self.shortcut(x)
        out = x + self.scale * out
        return out


# Reduction-A
class ReductionA(nn.Module):
    def __init__(self, in_channel):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.branch2 = BasicConv1d(in_channel=in_channel, out_channel=384, kernel_size=3, stride=2, padding=0)

        self.branch3_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = BasicConv1d(in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = BasicConv1d(in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        out1 = self.branch1(x)

        out2 = self.branch2(x)

        out3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))

        # print(out1.shape, out2.shape, out3.shape)
        return torch.cat((out1, out2, out3), dim=1)


# Inception-ResNet-B
class InceptionResNetB(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionResNetB, self).__init__()
        self.branch1 = BasicConv1d(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=128, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=128, out_channel=192, kernel_size=7, stride=1, padding=3)

        self.linear = BasicConv1d(in_channel=384, out_channel=1152, kernel_size=1, stride=1, padding=0)
        self.out_channel = 1152
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        output1 = self.branch1(x)

        output2 = self.branch2_2(self.branch2_1(x))
        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)

        x = self.shortcut(x)
        out = x + out * self.scale
        out = F.relu(out)
        return out


# Reduction-B
class ReductionB(nn.Module):
    def __init__(self, in_channel):
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=256, out_channel=384, kernel_size=3, stride=2, padding=0)

        self.branch3_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = BasicConv1d(in_channel=256, out_channel=288, kernel_size=3, stride=2, padding=0)

        self.branch4_1 = BasicConv1d(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1, padding=0)
        self.branch4_2 = BasicConv1d(in_channel=256, out_channel=288, kernel_size=3, stride=1, padding=1)
        self.branch4_3 = BasicConv1d(in_channel=288, out_channel=320, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        out1 = self.branch1(x)

        out2 = self.branch2_2(self.branch2_1(x))

        out3 = self.branch3_2(self.branch3_1(x))

        out4 = self.branch4_3(self.branch4_2(self.branch4_1(x)))

        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        return torch.cat((out1, out2, out3, out4), dim=1)


# Inception-ResNet-C
class InceptionResNetC(nn.Module):
    def __init__(self, in_channel, scale=0.1):
        super(InceptionResNetC, self).__init__()
        self.branch1 = BasicConv1d(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)

        self.branch2_1 = BasicConv1d(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = BasicConv1d(in_channel=192, out_channel=256, kernel_size=3, stride=1, padding=1)

        self.linear = BasicConv1d(in_channel=448, out_channel=2144, kernel_size=1, stride=1, padding=0)
        self.out_channel = 2144
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        output1 = self.branch1(x)

        output2 = self.branch2_2(self.branch2_1(x))

        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)

        x = self.shortcut(x)
        out = x + out * self.scale
        out = F.relu(out)
        return out


# Inception_ResNet_v2
class InceptionResNetV2(nn.Module):
    def __init__(self):
        super(InceptionResNetV2, self).__init__()
        # stem
        blocks = [Stem(n_channels)]
        # Inception-ResNet-A
        for _ in range(inception_A_block_num):
            blocks.append(InceptionResNetA(384))
        # Reduction-A
        blocks.append(ReductionA(384))
        # Inception-ResNet-B
        for _ in range(inception_B_block_num):
            blocks.append(InceptionResNetB(1152))
        # Reduction-B
        blocks.append(ReductionB(1152))
        # Inception-ResNet-C
        for _ in range(inception_C_block_num):
            blocks.append(InceptionResNetC(2144))

        self.map = nn.Sequential(*blocks)
        self.pool = nn.AvgPool1d(kernel_size=8)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2144*2, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.map(x)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.linear(out.view(-1, 2144*2))
        out = self.softmax(out)
        return out


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1].cpu().numpy()
    y_label = torch.max(labels, 1)[1].data.cpu().numpy()
    rights = (pred == y_label).sum()
    return rights, len(labels)


n_channels = 2
n_classes = 7
inception_A_block_num = 5
inception_B_block_num = 10
inception_C_block_num = 5

if __name__ == '__main__':
    a = torch.randn(64, 2, 600)
    net = InceptionResNetV2()
    print(net(a).shape)
