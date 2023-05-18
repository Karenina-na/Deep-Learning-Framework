import torch
from Models.ResNet.Block.ResBottlrneck_3conv_1d import ResBottlrneck as Bottlrneck

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_classes = 7
n_channels = 2


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=n_channels, classes=n_classes):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            #
            Bottlrneck(256, 128, 512, True),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            #
            Bottlrneck(512, 256, 1024, True),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            #
            Bottlrneck(1024, 512, 2048, True),
            Bottlrneck(2048, 512, 2048, False),
            Bottlrneck(2048, 512, 2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048, classes),
            torch.nn.Dropout(p=0.2, inplace=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 2048)
        x = self.classifer(x)
        return x


if __name__ == "__main__":
    x = torch.randn(64, 2, 600).to(device)
    model = ResNet().to(device)
    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
