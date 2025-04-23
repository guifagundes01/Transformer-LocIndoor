import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=4, num_layers=1, dropout=0.0) -> None:
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, birectional=True, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        x, _ = self.rnn(x)
        return self.fc(x)

