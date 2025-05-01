# import torch
from torch import nn, Tensor


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size=4, num_layers=1, dropout=0.0) -> None:
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, birectional=True, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2*hidden_size, output_size)
        # self.softmax = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x = self.fc(x)
        # return torch.cat((x[:,0:2], self.softmax(x[:, 0:2])), 1)
        return x

