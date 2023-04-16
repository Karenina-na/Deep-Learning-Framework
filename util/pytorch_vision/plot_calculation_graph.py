import torchvision.models as models
import torch
from tensorboard import notebook
from torch.utils.tensorboard import SummaryWriter


# 计算图可视化
def plot_calculation_graph(Net, tensor, path="../../Result/images/tensorboard"):
    writer = SummaryWriter(path)
    writer.add_graph(Net, input_to_model=tensor)
    writer.close()
    notebook.list()
    notebook.start("--logdir " + path)


if __name__ == '__main__':
    net = models.resnet18()
    plot_calculation_graph(net, torch.rand(1, 3, 224, 224))
    # 在命令行中运行
    # tensorboard --logdir ./Result/images/tensorboard
