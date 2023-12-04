import torch
import torch.nn as nn
from torch import Tensor


class PConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, n_div: int = 4, forward: str = 'split_cat'):
        super(PConv, self).__init__()
        assert in_channels > 4, "in_channels should > 4, but got {} instead.".format(in_channels)
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv

        self.conv = nn.Conv2d(in_channels=self.dim_conv, out_channels=self.dim_conv,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing

        elif forward == 'split_cat':
            self.forward = self.forward_split_cat

        else:
            raise NotImplementedError("forward method: {} is not implemented.".format(forward))

    def forward_slicing(self, x: Tensor) -> Tensor:
        p_conv = self.conv(x[:, :self.dim_conv, :, :])
        oth_conv = self.identity(x[:, self.dim_conv:, :, :])
        x = torch.cat((p_conv, oth_conv), dim=1)
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), dim=1)

        return x
