# -*- coding:utf8 -*-
import torch
import torch.nn.functional as F
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class Drop(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7):
        super(Drop, self).__init__()

        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        if self.drop_rate == 0:
            return x
        # 设置gamma,比gamma小的设置为1,大于gamma的为0,对应第五步
        gamma = self.drop_rate / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)
        # apply block mask,为算法图的第六步
        out = x * block_mask[:, None, :, :]
        # Normalize the features,对应第七步
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


if __name__ == '__main__':
    drop = Drop(drop_rate=0.9, block_size=2)
    x = torch.randn(1, 1, 5, 5)
    y = drop(x)
    print(x)
    print(y)
