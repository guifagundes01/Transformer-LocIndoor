from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence


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
                 padding_id=0, output_dim=2, num_layers=1, dropout=0.0) -> None:
        super(RNNRegressorEmb, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_id)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=True,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2*hidden_size, output_dim)

    def forward(self, x: Tensor, lenghts: Tensor) -> Tensor:
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lenghts.cpu(), batch_first=True)
        _, (h_n, _) = self.rnn(packed)
        return self.fc(h_n[-1])

