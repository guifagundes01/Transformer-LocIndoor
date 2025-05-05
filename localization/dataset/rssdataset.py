from typing import Optional

import numpy as np
import polars as pl

from numpy.typing import NDArray
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.utils.data.dataset import Dataset

class RSSDataset(Dataset):
    def __init__(self, file_path: str, device: DeviceLikeType, routers: Optional[NDArray[np.str_]] = None,
                 cols2drop: Optional[list[str]] = None, y_cols: Optional[list[str]] = None):
        if cols2drop is None:
            cols2drop = ["x", "y", "BUILDINGID", "FLOOR"]
        if y_cols is None:
            y_cols = ["x", "y"]

        x = pl.read_csv(file_path).unique()
        if routers is None:
            routers_cols = list(set(x.columns) - set(["x", "y", "BUILDINGID", "FLOOR"]))
            mask = ((x[routers_cols] == 0).sum() != x.shape[0]).to_numpy()[0]
            self.routers: NDArray[np.str_] = np.array(routers_cols)[mask]
        else:
            self.routers = routers

        self.y = x[y_cols].to_numpy()
        self.x = x.drop(cols2drop)[self.routers].to_numpy()
        self.device = device

    def __getitem__(self, index: int):
        x = Tensor(self.x[index]).to(self.device)
        y = Tensor(self.y[index]).to(self.device)
        return (x, y)

    def __len__(self):
        return self.x.shape[0]
