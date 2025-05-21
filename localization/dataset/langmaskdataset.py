import random
from typing import Tuple

import torch
import numpy as np
from torch.nn.functional import pad
from torch.utils.data.dataset import Dataset

class LangMaskDataset(Dataset):
    def __init__(self, file_path: str, device, src_dim: int, vocab_size: int, sos_idx: int, padding_idx = 0, prob = 0.4):
        data = torch.load(file_path, weights_only=False)
        seqs, _ = zip(*data)
        self.sequences = torch.empty((len(seqs), src_dim), dtype=torch.long)
        tars = []
        ids = set(range(vocab_size))
        for i, seq in enumerate(seqs):
            if random.random() < prob:
                available_ids = list(ids - set(seq))
                if available_ids:
                    new_id = random.choice(available_ids)
                    insert_pos = random.randint(0, len(seq))
                    np.insert(seq, insert_pos, new_id)
                    tars.append(new_id)
                else:
                    tars.append(vocab_size)
            else:
                tars.append(vocab_size)

            seq = torch.tensor(seq, dtype=torch.long)
            seq = pad(seq + 1, (1, 0), value=sos_idx)
            if seq.size(0) < src_dim:
                seq = pad(seq, (0, src_dim - seq.size(0)), value=padding_idx)
            else:
                seq = seq[:src_dim]
            self.sequences[i] = seq

        self.sequences = self.sequences.to(device)
        self.targets = torch.tensor(np.array(tars), dtype=torch.long).to(device)

    def __getitem__(self, index: int)  -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.targets[index]

    def __len__(self):
        return len(self.targets)
