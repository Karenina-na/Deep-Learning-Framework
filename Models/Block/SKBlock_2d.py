import torch.nn as nn
import torch


class SK(nn.Module):
    def __init__(self, input_channel, branchs_number=3, conv_group=1, ratio=4, stride=1, L=32) -> None:

        super().__init__()
        d = max(int(input_channel / ratio), L)  # 用来进行线性层的输出通道，input_channel，用L就有点丢失数据了。
        self.branchs_number = branchs_number
        self.input_channel = input_channel
        self.convs = nn.ModuleList([])
        for i in range(branchs_number):
            self.convs.append(
                nn.Sequential(
                    # 分组卷积
                    nn.Conv2d(input_channel, input_channel, kernel_size=3 + i * 2, stride=stride, padding=1 + i,
                              groups=conv_group),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU(inplace=True)
                )
            )
        self.fc = nn.Linear(input_channel, d)
        self.fcs = nn.ModuleList([])
        for i in range(branchs_number):
            self.fcs.append(nn.Linear(d, input_channel))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 第一部分，每个分支的数据进行相加,虽然这里使用的是torch.cat，但是后面又用了unsqueeze和sum进行升维和降维
        for i, conv in enumerate(self.convs):
            fea = conv(x).clone().unsqueeze_(dim=1).clone()  # 这里在1这个地方新增了一个维度  [batch, 1, channel, W, H]
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas.clone(), fea], dim=1)  # feas.shape [batch, branchs, channel, W, H]
        fea_U = torch.sum(feas.clone(), dim=1)  # [batch, channel, W, H] # 这里是对每个分支的数据进行相加
        fea_s = fea_U.clone().mean(-1).mean(-1)  # [batch, channel]
        fea_z = self.fc(fea_s)  # [batch, channel] -> [batch, d]
        # 第二部分，对每个分支的数据进行softmax操作
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).clone().unsqueeze_(dim=1)  # [batch, d] -> [batch, channel] -> [batch, 1, channel]
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors.clone(), vector], dim=1)  # 同样的相加操作 # [batch, branchs, channel]
        attention_vectors = self.softmax(attention_vectors.clone())  # 对每个分支的数据进行softmax操作
        attention_vectors = attention_vectors.clone().unsqueeze(-1).unsqueeze(-1)  # ->[batch, branchs, channel, 1, 1]
        fea_v = (feas * attention_vectors).clone().sum(dim=1)  # ->[batch, channel, W, H]
        return fea_v


if __name__ == "__main__":
    x = torch.randn(16, 64, 128, 128)
    # input_channel 数据输入维度，branchs_number为分支数，
    # conv_group为Conv2d层的组数，基本设置为1，ratio用来进行求线性层输出通道的，缩放比例。
    sk = SK(input_channel=64, branchs_number=3, conv_group=1, ratio=2)
    out = sk(x)
    print(out.shape)
