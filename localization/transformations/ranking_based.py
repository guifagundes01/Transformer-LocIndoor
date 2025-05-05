from typing import Any

import numpy as np
from numpy._typing import NDArray

class RankingBased:
    # def __init__(self):
    #     None

    def transform(self, X: NDArray[Any], null_element=0, default_value=-1):
        return np.apply_along_axis(lambda row: self.get_ranking(row, null_element, default_value), 1, X)

    def get_ranking(self, rss: NDArray[Any], null_element=0, default_value=-1):
        # the strongest to the weakest AP
        ix = np.argsort(rss)[::-1]
        # index of first null element
        first_null = rss.shape[0]
        null_ix = np.argwhere(rss[ix] == null_element)
        if len(null_ix) > 0:
            first_null = int(null_ix[0])
        ranking = np.zeros(len(ix)) + default_value
        ranking[ix[:first_null]] = np.arange(1, first_null + 1)
        return ranking
