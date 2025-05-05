import argparse

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from localization import utils
from localization.models import RNNRegressor


def transform(df, input_columns):
    # Sanity check
    if any(df[input_columns[0]] == 100):
        df = df.replace(100, -105)
        df[input_columns] = df[input_columns] + 105
    return df.copy()

def calculate_means(dataframe):
    return [dataframe['LATITUDE'].mean(), dataframe['LONGITUDE'].mean()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN Model')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-f', '--data_folder', type=str, default="data/generated", help='Data folder')
    parser.add_argument('-d', '--out_dir', type=str, default="output/rnn", help='Output folder')
    parser.add_argument('-r', '--r_path', type=str, default="data/routers.npy", help='File to used routers')

    args = parser.parse_args()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    utils.make_deterministic(args.seed)

    routers = np.load(args.r_path)
    dfv = pd.read_csv(f"{args.data_folder}/validationData.csv")
    dft = pd.read_csv(f"{args.data_folder}/trainingData.csv")
    input_columns = list(dft.filter(regex=r'WAP\d+').columns)

    dft = transform(dft, input_columns)
    dfv = transform(dfv, input_columns)

    buildings = dft['BUILDINGID'].unique()
    for building in buildings:
        building_indexes_train = dft['BUILDINGID']==building
        building_indexes_validation = dfv['BUILDINGID']==building
        means = calculate_means(dft[building_indexes_train])
        dft.loc[building_indexes_train,'x'] = dft[building_indexes_train]['LATITUDE'] - means[0]
        dft.loc[building_indexes_train,'y'] = dft[building_indexes_train]['LONGITUDE'] - means[1]
        dfv.loc[building_indexes_validation,'x'] = dfv[building_indexes_validation]['LATITUDE'] - means[0]
        dfv.loc[building_indexes_validation,'y'] = dfv[building_indexes_validation]['LONGITUDE'] - means[1]

    df = pd.concat([dft, dfv])
    x_test = df[routers].to_numpy()
    y_test = df[["x", "y"]].to_numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNRegressor(x_test.shape[1], 256, 2).to(device)
    model.load_state_dict(torch.load(f"{args.out_dir}/rnn_model.pth", map_location=device))

    input_tensor = torch.Tensor(x_test).to(device)

    # model.eval()
    with torch.no_grad():
        y_pred = model(input_tensor)

    y_pred = y_pred.cpu().numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
