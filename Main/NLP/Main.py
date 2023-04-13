import torch
import torch.nn as nn
import torchtext

# 德语到英语的翻译数据集
torchtext.datasets.Multi30k(root='./Data/', split=('train', 'valid', 'test'), language_pair=('de', 'en'))
