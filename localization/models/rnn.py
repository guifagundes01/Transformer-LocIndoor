import torch
from torch import IntTensor, nn, Tensor


class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim=2, num_layers=1, dropout=0.0) -> None:
        super(RNNRegressor, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, bidirectional=True,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2*hidden_size, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


class RNNRegressorEmb(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 output_dim=2, padding_id=0, num_layers=1, dropout=0.0) -> None:
        super(RNNRegressorEmb, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_id)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=True,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2*hidden_size, output_dim)
        # self.fc = nn.Sequential(
        #         nn.Linear(2*hidden_size, hidden_size // 4),
        #         nn.ReLU(),
        #         nn.Linear(hidden_size // 4, output_dim)
        # )

    # def forward(self, x: IntTensor, lenghts: Tensor) -> Tensor:
    #     embedded = self.embedding(x)
    #     packed = nn.utils.rnn.pack_padded_sequence(embedded, lenghts.cpu(), batch_first=True, enforce_sorted=False)
    #     _, (h_n, _) = self.rnn(packed)
    #     h_nc = torch.cat((h_n[0], h_n[1]), dim=1)
    #     return self.fc(h_nc)

    def forward(self, x: IntTensor) -> Tensor:
        embedded = self.embedding(x)
        _, (h_n, _) = self.rnn(embedded)
        h_nc = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h_nc)

