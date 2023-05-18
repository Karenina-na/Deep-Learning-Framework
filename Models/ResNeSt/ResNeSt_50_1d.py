import torch
from Models.ResNeSt.Block.DropBlock_1d import Drop as DropBlock_1d
from Models.ResNeSt.Block.ResNeStBlock_1d import ResNeStBlock_1d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNeSt(torch.nn.Module):
    def __init__(self):
        super(ResNeSt, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            ResNeStBlock_1d(in_channels=64, med_channels=64, out_channels=256, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=256, med_channels=64, out_channels=256, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=256, med_channels=64, out_channels=256, cardinality=2, radix=4,
                            down_sample=False),
            #
            DropBlock_1d(drop_rate=0.2, block_size=7),
            #
            ResNeStBlock_1d(in_channels=256, med_channels=128, out_channels=512, cardinality=2, radix=4,
                            down_sample=True),
            ResNeStBlock_1d(in_channels=512, med_channels=128, out_channels=512, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=512, med_channels=128, out_channels=512, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=512, med_channels=128, out_channels=512, cardinality=2, radix=4,
                            down_sample=False),
            #
            DropBlock_1d(drop_rate=0.2, block_size=7),
            #
            ResNeStBlock_1d(in_channels=512, med_channels=256, out_channels=1024, cardinality=2, radix=4,
                            down_sample=True),
            ResNeStBlock_1d(in_channels=1024, med_channels=256, out_channels=1024, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=1024, med_channels=256, out_channels=1024, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=1024, med_channels=256, out_channels=1024, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=1024, med_channels=256, out_channels=1024, cardinality=2, radix=4,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=1024, med_channels=256, out_channels=1024, cardinality=2, radix=4,
                            down_sample=False),
            #
            DropBlock_1d(drop_rate=0.2, block_size=7),
            #
            ResNeStBlock_1d(in_channels=1024, med_channels=512, out_channels=2048, cardinality=4, radix=8,
                            down_sample=True),
            ResNeStBlock_1d(in_channels=2048, med_channels=512, out_channels=2048, cardinality=4, radix=8,
                            down_sample=False),
            ResNeStBlock_1d(in_channels=2048, med_channels=512, out_channels=2048, cardinality=4, radix=8,
                            down_sample=False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(2048, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 2048)
        x = self.classifer(x)
        return x


n_channels = 2
n_classes = 7

if __name__ == '__main__':
    a = torch.randn(64, 2, 600)
    net = ResNeSt()
    print(net(a).shape)
