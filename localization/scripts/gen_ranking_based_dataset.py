import argparse
import numpy as np
import pandas as pd
from localization.transformations.ranking_based import RankingBased 
from os import mkdir, path

def convert_to_ranking_based(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    
    wap_columns = [col for col in df.columns if col.startswith('WAP')]
    df_rss = df[wap_columns]

    ranker = RankingBased(wap_columns, null_element=0.0)
    df_sequences = ranker.transform(df_rss)

    non_wap_columns = df.drop(columns=wap_columns)
    df_final = pd.concat([df_sequences, non_wap_columns.reset_index(drop=True)], axis=1)

    if not path.exists(output_dir): mkdir(output_dir)
    df_final.to_csv(f'{output_dir}/rankingBasedData.csv', index=False)

if __name__ == "__main__":
    convert_to_ranking_based(
        input_csv="data/generated/trainingData.csv",
        output_dir="data/ranking_based/"
    )