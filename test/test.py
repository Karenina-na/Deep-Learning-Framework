import os
import torch
from torch import nn
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

epoch = 2
batch_size = 64
time_step = 28  # 时间步数（图片高度）（因为每张图像为28*28，而每一个序列长度为1*28，所以总共是28个1*28）
input_size = 28  # 每步输入的长度（每行像素的个数）
lr = 0.01
download_mnist = True

num_classes = 10  # 总共有10类
hidden_size = 128  # 隐层大小
num_layers = 1

# MINIST
train_data = torchvision.datasets.MNIST(
    root='../Data/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=download_mnist,
)

test_data = torchvision.datasets.MNIST(
    root='../Data/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=download_mnist,
)

# print(train_data.train_data.size())  # [60000, 28, 28]
# print(train_data.train_labels.size())  # [60000]
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title("MNIST:%i" % train_data.train_labels[0])
# plt.show()

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

data = next(iter(train_loader))  # train_loader是迭代器
# print(data[0].shape)  # [64, 1, 28, 28] 64张图片，1为通道数（灰白图片），28*28为图片大小
# print(data[1].shape)  # [64] 64张图片的标签

# 遍历数据
for step, (b_x, b_y) in enumerate(train_loader):  # 遍历数据和标签
    print(b_x.shape)  # [64, 1, 28, 28]
    b_x = b_x.view(-1, 28, 28)  # 将tensor拉成(64*28*28)
    print(b_x.shape)  # [64, 28, 28]
    print(b_y.shape)  # [64]

    print(b_x[0].shape)  # [28, 28]
    print(b_y[0])  # [1]
    break


