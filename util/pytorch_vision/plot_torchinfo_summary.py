import torchvision.models as models
from torchinfo import summary
import torch


# torchinfo
def torchinfo_summary(net, tensor):
    """
    :param net: 网络
    :param tensor: 输入
    """
    summary(net, input_data=tensor)


if __name__ == '__main__':
    model = models.resnet18()
    torchinfo_summary(model, torch.rand(1, 3, 224, 224))
