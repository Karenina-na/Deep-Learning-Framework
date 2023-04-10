import torch


class ForgetGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ForgetGate, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化forget gate的参数
        self.w_if = torch.nn.Linear(input_size, hidden_size)
        self.w_hf = torch.nn.Linear(hidden_size, hidden_size)
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input, ht):
        """
        :param input: 输入
        :param ht: 上一时刻的隐层输出
        :return: ft
        """

        # 计算forget gate的输出, ft=sigmoid(W_if*xt+W_hf*ht+b_f)
        ft = torch.sigmoid(self.w_if(input) + self.w_hf(ht) + self.b_f)

        # 返回
        return ft


class InputGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputGate, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化input gate的参数
        self.w_ii = torch.nn.Linear(input_size, hidden_size)
        self.w_hi = torch.nn.Linear(hidden_size, hidden_size)
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))

        self.w_ic = torch.nn.Linear(input_size, hidden_size)
        self.w_hc = torch.nn.Linear(hidden_size, hidden_size)
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input, ht):
        """
        :param input: 输入
        :param ht: 上一时刻的隐层输出
        :return: it
        """

        # 计算input gate的输出, it=sigmoid(W_ii*xt+W_hi*ht+b_i)
        it = torch.sigmoid(self.w_ii(input) + self.w_hi(ht) + self.b_i)

        # 计算input gate的输出, ct=tanh(W_ic*xt+W_hc*ht+b_c)
        ct = torch.tanh(self.w_ic(input) + self.w_hc(ht) + self.b_c)

        # 返回
        return it, ct


class InputOutputGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputOutputGate, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化output gate的参数
        self.w_io = torch.nn.Linear(input_size, hidden_size)
        self.w_ho = torch.nn.Linear(hidden_size, hidden_size)
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, cell, input, ht):
        """
        :param cell: 更新后的cell
        :param input: 输入
        :param ht: 上一时刻的隐层输出
        :return:
        """
        # 计算变换输出, ot=sigmoid(W_io*xt+W_ho*ht+b_o)
        ot = torch.sigmoid(self.w_io(input) + self.w_ho(ht) + self.b_o)

        # 计算下一时刻信息, ht=ot*tanh(cell)
        ht = ot * torch.tanh(cell)

        return ht


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 存储状态
        self.ht = torch.zeros(1, hidden_size)
        self.cell = torch.zeros(1, hidden_size)

        # 初始化forget gate
        self.forget_gate = ForgetGate(input_size, hidden_size)

        # 初始化input gate
        self.input_gate = InputGate(input_size, hidden_size)

        # 初始化output gate
        self.output_gate = InputOutputGate(input_size, hidden_size)

        # 预测输出层
        self.predict = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        :param input: 输入
        :return: 下一时刻的隐层输出和cell
        """
        # 计算forget gate的输出
        ft = self.forget_gate(input, self.ht)

        # 计算input gate的输出
        it, ct = self.input_gate(input, self.ht)

        # 计算cell
        self.cell = ft * self.cell + it * ct

        # 计算下一时刻的隐层输出
        self.ht = self.output_gate(self.cell, input, self.ht)

        # 预测输出概率
        return torch.softmax(self.predict(self.ht), dim=1)

    def reset(self):
        self.ht = torch.zeros(1, self.hidden_size)
        self.cell = torch.zeros(1, self.hidden_size)


if __name__ == '__main__':
    # 输入数据 [batch_size, sel_len, [word_embedding]]
    input = torch.randn(64, 10, 32)

    # 初始化模型
    model = LSTM(input_size=32, hidden_size=64, output_size=32)

    # 模拟训练
    for i in range(input.size(1)):
        # 训练
        output = model(input[:, i, :])
        print(output.shape)

    # 重置状态
    model.reset()
