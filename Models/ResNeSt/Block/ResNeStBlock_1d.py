import torch
import torch.nn as nn
import torch.nn.functional as F


# conv + bn + relu
class BasicConv1d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, ):
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


# Split attention
class SplAtConv_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, radix):
        super(SplAtConv_1d, self).__init__()
        self.radix = radix
        self.groups = groups

        self.radix_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * radix, kernel_size, stride, padding, groups=groups * radix),
            nn.BatchNorm1d(out_channels * radix),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Conv1d(out_channels, out_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Conv1d(out_channels, out_channels * radix, 1, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, X):
        """
        input  : |             out_channels * radix               |
        """
        '''
        radix_conv : |                radix 0            |               radix 1             | ... |                radix r            |
                     | group 0 | group 1 | ... | group k | group 0 | group 1 | ... | group k | ... | group 0 | group 1 | ... | group k |
        '''
        X = self.radix_conv(X)

        '''
        按 radix 分组，每个split通道数为 out_channels
        splits :  [ | group 0 | group 1 | ... | group k |,  | group 0 | group 1 | ... | group k |, ... ]
        '''
        splits = torch.split(X, X.shape[1] // self.radix, dim=1)

        '''
        按 group[i] 求和, 得到每个group[i]的和
        sum   :  | group 0 | group 1 | ...| group k |
        '''
        gap = sum(splits)
        '''
        求每个group[i]的平均值
        gap   :  | group 0 | group 1 | ...| group k |
        '''
        gap = self.avg_pool(gap)
        '''
        线性映射
        '''
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        '''
        注意力
        attention : |                radix 0            |               radix 1             | ... |                radix r            |
                    | group 0 | group 1 | ... | group k | group 0 | group 1 | ... | group k | ... | group 0 | group 1 | ... | group k |
        '''
        attention = self.fc2(gap)
        attention = self.rsoftmax(attention).view(X.shape[0], -1, 1)
        '''
        按 radix 分组，每个attentions通道数为 out_channels
        attentions :  [ | group 0 | group 1 | ... | group k |,  | group 0 | group 1 | ... | group k |, ... ]
        '''
        attentions = torch.split(attention, attention.shape[1] // self.radix, dim=1)

        # 将 gap 通道计算出的注意力乘原始分出的 radix 组，并将对应 group[i] 相加得到最后结果
        '''
        out         |      radix 0 * attentions 0       |      radix 1 * attentions 1       |      radix 2 * attentions 2       |
                    | group 0 | group 1 | ... | group k | group 0 | group 1 | ... | group k | group 0 | group 1 | ... | group k |
        '''
        out = sum([att * split for (att, split) in zip(attentions, splits)])
        # 返回一个 out 的copy, 使用 contiguous 是保证存储顺序的问题
        return out.contiguous()


# 对radix维度进行softmax
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, X):
        batch = X.size(0)

        X = X.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        # x: [batch, radix, cardinality, h, w]
        X = F.softmax(X, dim=1)  # 对radix维度进行softmax
        X = X.reshape(batch, -1)
        return X


# ResNeSt Block
class ResNeStBlock_1d(nn.Module):
    def __init__(self, in_channels, med_channels, out_channels, cardinality, radix, down_sample=False):
        super(ResNeStBlock_1d, self).__init__()
        self.stride = 1
        if down_sample:
            self.stride = 2

        # 1x1卷积
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, med_channels, 1, self.stride),
            torch.nn.BatchNorm1d(med_channels),
            torch.nn.ReLU(),
        )

        # SplAtConv_1d
        self.SplAtConv = SplAtConv_1d(med_channels, med_channels, 3, 1, 1, cardinality, radix)

        # 1x1卷积
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(med_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
        )

        if in_channels != out_channels:
            self.res_layer = torch.nn.Conv1d(in_channels, out_channels, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, X):
        # 1x1卷积
        out = self.conv1(X)
        # SplAtConv
        out = self.SplAtConv(out)
        # 1x1卷积
        out = self.conv2(out)

        # 残差连接
        if self.res_layer is not None:
            return out + self.res_layer(X)
        else:
            return out + X


if __name__ == '__main__':
    x = torch.randn(128, 16, 600)
    # model = SplAtConv_1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, groups=1, radix=2)
    model = ResNeStBlock_1d(in_channels=16, med_channels=64, out_channels=128, cardinality=2, radix=4, down_sample=True)
    y = model(x)
    print(y.shape)
