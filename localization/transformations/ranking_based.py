import numpy as np
import math
import pandas as pd
import string
import math

class RankingBased:
    def __init__(self, wap_list, null_element=0.0):
        self.wap_list = list(wap_list)
        self.vocab = self._build_vocab()
        self.null_element = null_element

    def _build_vocab(self):
        # Build token vocabulary using alphanumeric tokens
        # base_chars = string.ascii_letters + string.digits
        # vocab_size = len(self.wap_list)
        # tokens = []
        # i = 0
        # while len(tokens) < vocab_size:
        #     for c in base_chars:
        #         token = base_chars[i // len(base_chars)] + c if i >= len(base_chars) else c
        #         tokens.append(token)
        #         if len(tokens) == vocab_size:
        #             break
        #     i += 1
        # return dict(zip(self.wap_list, tokens))

        # For simplicity and visualization, use the router names as tokens
        return {router: f"W{router[3:]}" for idx, router in enumerate(self.wap_list)}

    def rss_row_to_sequence(self, row):
        row = np.array(row)
        indices = np.argsort(row)[::-1]  # strongest to weakest
        sequence = []
        for idx in indices:
            if not math.isclose(row[idx], self.null_element, abs_tol=1e-5):
                wap_name = self.wap_list[idx]
                token = self.vocab[wap_name]
                sequence.append(token)
        return sequence

    def transform(self, df_rss):
        sequences = [self.rss_row_to_sequence(row) for row in df_rss.values]
        return pd.DataFrame({"sequence": sequences})

    def get_vocab(self):
        return self.vocab
    

# Deprecated code - Keeping the same size and putting -1 in the null elements
# class RankingBased:
#     def __init__(self):
#         None

#     def transform(self, X, null_element=0, default_value=-1):
#         return np.apply_along_axis(lambda row: self.get_ranking(row, null_element, default_value), 1, X)

#     def get_ranking(self, rss, null_element, default_value=-1):
#         # the strongest to the weakest AP
#         ix = np.argsort(rss)[::-1]
#         # index of first null element
#         first_null = rss.shape[0]
#         null_ix = np.argwhere(rss[ix] == null_element)
#         if len(null_ix) > 0:
#             first_null = int(null_ix[0])
#         ranking = np.zeros(len(ix)) + default_value
#         ranking[ix[:first_null]] = np.arange(1, first_null + 1)
#         return ranking
