from typing import List, Tuple, Any

import torch
from numpy.typing import NDArray
from torch.utils.data.dataset import Dataset

class LangDataset(Dataset):
    def __init__(self, file_path: str, device):
        self.data: List[Tuple[NDArray[Any], NDArray[Any]]] = torch.load(file_path, weights_only=False)
        self.device = device

    def __getitem__(self, index: int):
        ins, out = self.data[index]
        seq = torch.tensor(ins, dtype=torch.int32).to(self.device)
        target = torch.tensor(out, dtype=torch.float32).to(self.device)
        return seq + 1, target

    def __len__(self):
        return len(self.data)
