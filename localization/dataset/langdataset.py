from typing import List, Tuple, Any

import torch
from numpy.typing import NDArray
from torch.nn.functional import pad
from torch.utils.data.dataset import Dataset

class LangDataset(Dataset):
    def __init__(self, file_path: str, device, src_dim: int, sos_idx: int, padding_idx = 0):
        self.data: List[Tuple[NDArray[Any], NDArray[Any]]] = torch.load(file_path, weights_only=False)
        self.device = device
        self.padding_idx = padding_idx
        self.sos_idx = sos_idx
        self.src_dim = src_dim

    def __getitem__(self, index: int):
        seq, target = self.data[index]
        seq = torch.tensor(seq, dtype=torch.int32).to(self.device)
        target = torch.tensor(target, dtype=torch.float32).to(self.device)
        seq = torch.cat((torch.tensor(self.sos_idx), seq+1))
        if seq.shape[0] < self.src_dim:
            seq = pad(seq, (0, self.src_dim - seq.shape[0]), "constant", self.padding_idx)
        else:
            seq = seq[:self.src_dim]
        return seq, target

    def __len__(self):
        return len(self.data)
