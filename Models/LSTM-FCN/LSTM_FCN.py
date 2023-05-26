import torch
import warnings
from collections import OrderedDict
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')


class LSTM(torch.nn.Module):

    def __init__(self, input_length, units, dropout):

        """
        Parameters:
        __________________________________
        input_length: int.
            Length of the time series.

        units: list of int.
            The length of the list corresponds to the number of recurrent blocks, the items in the
            list are the number of units of the LSTM layer in each block.

        dropout: float.
            Dropout rate to be applied after each recurrent block.
        """

        super(LSTM, self).__init__()

        # check the inputs
        if type(units) != list:
            raise ValueError(f'The number of units should be provided as a list.')

        # build the model
        modules = OrderedDict()
        for i in range(len(units)):
            modules[f'LSTM_{i}'] = torch.nn.LSTM(
                input_size=input_length if i == 0 else units[i - 1],
                hidden_size=units[i],
                batch_first=True
            )
            modules[f'Lambda_{i}'] = Lambda(f=lambda x: x[0])
            if i < len(units) - 1:
                modules[f'Dropout_{i}'] = torch.nn.Dropout(p=dropout)
        self.model = torch.nn.Sequential(modules)

    def forward(self, x):

        """
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, 1, length) where samples is the number of
            time series and length is the length of the time series.
        """

        return self.model(x)[:, -1, :]


class FCN(torch.nn.Module):

    def __init__(self, filters, kernel_sizes):
        """
        Parameters:
        __________________________________
        filters: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the number of filters (or channels) of the convolutional layer in each block.

        kernel_sizes: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the kernel sizes of the convolutional layer in each block.
        """

        super(FCN, self).__init__()

        # check the inputs
        if len(filters) == len(kernel_sizes):
            blocks = len(filters)
        else:
            raise ValueError(f'The number of filters and kernel sizes must be the same.')

        # build the model
        modules = OrderedDict()
        for i in range(blocks):
            modules[f'Conv1d_{i}'] = torch.nn.Conv1d(
                in_channels=1 if i == 0 else filters[i - 1],
                out_channels=filters[i],
                kernel_size=(kernel_sizes[i],),
                padding='same'
            )
            modules[f'BatchNorm1d_{i}'] = torch.nn.BatchNorm1d(
                num_features=filters[i],
                eps=0.001,
                momentum=0.99
            )
            modules[f'ReLU_{i}'] = torch.nn.ReLU()
        self.model = torch.nn.Sequential(modules)

    def forward(self, x):

        """
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, 1, length) where samples is the number of
            time series and length is the length of the time series.
        """

        return torch.mean(self.model(x), dim=-1)


class LSTM_FCN(torch.nn.Module):

    def __init__(self, input_length, units, dropout, filters, kernel_sizes, num_classes):
        """
        Parameters:
        __________________________________
        input_length: int.
            Length of the time series.

        units: list of int.
            The length of the list corresponds to the number of recurrent blocks, the items in the
            list are the number of units of the LSTM layer in each block.

        dropout: float.
            Dropout rate to be applied after each recurrent block.

        filters: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the number of filters (or channels) of the convolutional layer in each block.

        kernel_sizes: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the kernel sizes of the convolutional layer in each block.

        num_classes: int.
            Number of classes.
        """

        super(LSTM_FCN, self).__init__()

        self.fcn = FCN(filters, kernel_sizes)
        self.lstm = LSTM(input_length, units, dropout)
        self.linear = torch.nn.Linear(in_features=filters[-1] + units[-1], out_features=num_classes)

    def forward(self, x):
        """
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, length) where samples is the number of time series
            and length is the length of the time series.

        Returns:
        __________________________________
        y: torch.Tensor.
            Logits, tensor with shape (samples, num_classes) where samples is the number of time series
            and num_classes is the number of classes.
        """

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        y = torch.concat((self.fcn(x), self.lstm(x)), dim=-1)
        y = self.linear(y)

        return y


class Lambda(torch.nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


if __name__ == '__main__':
    x = np.random.rand(2, 100)

    model = LSTM_FCN(input_length=x.shape[1], units=[50, 50], dropout=0.2,
                     filters=[50, 60], kernel_sizes=[1, 1], num_classes=4)

    print(model)
