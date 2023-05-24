from functools import partial
from Models.FasterNet.FasterNet import FasterNet


def CreateFasterNetT0():
    return partial(FasterNet, inner_channels=[40, 80, 160, 320],
                   blocks=[1, 2, 8, 2], act='GELU', drop_path=0.)


def CreateFasterNetT1():
    return partial(FasterNet, inner_channels=[64, 128, 256, 512],
                   blocks=[1, 2, 8, 2], act='GELU', drop_path=0.02)


def CreateFasterNetT2():
    return partial(FasterNet, inner_channels=[96, 192, 384, 768],
                   blocks=[1, 2, 8, 2], act='ReLU', drop_path=0.05)


def CreateFasterNetS():
    return partial(FasterNet, inner_channels=[128, 256, 512, 1024],
                   blocks=[1, 2, 13, 2], act='ReLU', drop_path=0.1)


def CreateFasterNetM():
    return partial(FasterNet, inner_channels=[144, 288, 576, 1152],
                   blocks=[3, 4, 18, 3], act='ReLU', drop_path=0.2)


def CreateFasterNetL():
    return partial(FasterNet, inner_channels=[192, 384, 768, 1536],
                   blocks=[3, 4, 18, 3], act='ReLU', drop_path=0.3)
