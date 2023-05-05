import torch.nn as nn


def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.2)
        nn.init.constant_(m.bias, 0)
