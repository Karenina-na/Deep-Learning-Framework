import torch
from torchviz import make_dot
from torchvision.models import vgg16


def plot_model_graphviz(net, tensor, path="../../Result/images"):
    """
    画出模型的结构图
    :param net: 模型
    :param tensor: 输入的张量
    :param path: 保存路径
    :return:
    """
    g = make_dot(net(tensor), params=dict(net.named_parameters()))
    g.view(directory=path, cleanup=True)


if __name__ == '__main__':
    model = vgg16()  # 实例化 vgg16，网络可以改成自己的网络
    plot_model_graphviz(model, torch.randn(4, 3, 32, 32))
