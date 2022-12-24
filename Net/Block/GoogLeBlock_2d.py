import torch


class GoogLeBlock(torch.nn.Module):
    """inception模块"""

    def __init__(self, in_channels, conv1_out_channels,
                 conv2_Med_channels, conv2_out_channels,
                 conv3_Med_channels, conv3_out_channels,
                 pool_channels,
                 ):
        super(GoogLeBlock, self).__init__()

        # 1*1卷积层
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, conv1_out_channels, kernel_size=1),
            torch.nn.ReLU(inplace=True)
        )

        # 1*1卷积层+3*3卷积层
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, conv2_Med_channels, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(conv2_Med_channels, conv2_out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        # 1*1卷积层+5*5卷积层
        self.branch3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, conv3_Med_channels, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(conv3_Med_channels, conv3_out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
        )

        # 3*3最大池化层+1*1卷积层
        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels, pool_channels, kernel_size=1),
            torch.nn.ReLU(inplace=True),
        )

        self.ReLU = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        outputs = torch.cat((x1, x2, x3, x4), dim=1)
        return outputs


G = GoogLeBlock(3, 64, 96, 128, 16, 32, 32)
data = torch.randn(1, 3, 224, 224)
print(G(data).shape)
