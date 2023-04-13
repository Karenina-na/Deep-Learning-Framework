import torch


class ResBlock(torch.nn.Module):
    """一维残差块，两个卷积层"""
    def __init__(self, In_channel, Med_channel, Out_channel, DownSample=False):
        super(ResBlock, self).__init__()
        self.stride = 1
        if DownSample:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(In_channel, Med_channel, 3, self.stride, padding=1),
            torch.nn.BatchNorm2d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Med_channel, Out_channel, 3, padding=1),
            torch.nn.BatchNorm2d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv2d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual
