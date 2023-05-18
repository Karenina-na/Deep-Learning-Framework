import torch
from Models.ResNet.Block.ResBlock_2conv_1d import ResBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_classes = 7
n_channels = 2


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=n_channels, classes=n_classes):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            #
            ResBlock(64, 128, 128, True),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            #
            ResBlock(128, 256, 256, True),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            #
            ResBlock(256, 512, 512, True),
            ResBlock(512, 512, 512, False),
            ResBlock(512, 512, 512, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512, classes),
            torch.nn.Dropout(p=0.1, inplace=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifer(x)
        return x


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1].cpu().numpy()
    y_label = torch.max(labels, 1)[1].data.cpu().numpy()
    rights = (pred == y_label).sum()
    return rights, len(labels)


def test():
    x = torch.randn(708, 2, 600).to(device)
    model = ResNet().to(device)
    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')


if __name__ == "__main__":
    test()
