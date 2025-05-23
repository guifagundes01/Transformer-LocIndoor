# Author: Alexandre Maranhão <alexandremr01@gmail.com>
from typing import Tuple
from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from numpy.typing import NDArray


def load_ujiindoor_loc(data_folder='data', transform=True):
    df_training = pd.read_csv(path.join(data_folder, 'trainingData.csv'))
    df_validation = pd.read_csv(path.join(data_folder, 'validationData.csv'))
    return UJIIndoorLoc(df_training, df_validation, transform=transform)


def calculate_means(dataframe):
    return [dataframe['LATITUDE'].mean(), dataframe['LONGITUDE'].mean()]


class UJIIndoorLoc():
    def __init__(self, df_training, df_validation, transform):
        self.df_training, self.df_validation, self.input_columns = self.preprocess(df_training, df_validation,
                                                                                   transform)
        self.location_columns = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']

    def preprocess(self, df_training: pd.DataFrame, df_validation, transform=True, mix_datasets=False):

        if mix_datasets and transform:
            df = pd.concat([df_training, df_validation])
            df_training, df_validation = train_test_split(df, test_size=0.2, random_state=0)

        input_columns = list(df_training.filter(regex=r'WAP\d+').columns)

        df_training = self.drop_rows_without_information(df_training, input_columns)
        df_validation = self.drop_rows_without_information(df_validation, input_columns)
        if transform:
            df_training = self.transform(df_training, input_columns)
            df_validation = self.transform(df_validation, input_columns)

            buildings = df_training['BUILDINGID'].unique()
            for building in buildings:
                building_indexes_train = df_training['BUILDINGID']==building
                building_indexes_validation = df_validation['BUILDINGID']==building
                means = calculate_means(df_training[building_indexes_train])
                df_training.loc[building_indexes_train,'x'] = df_training[building_indexes_train]['LATITUDE'] - means[0]
                df_training.loc[building_indexes_train,'y'] = df_training[building_indexes_train]['LONGITUDE'] - means[1]
                df_validation.loc[building_indexes_validation,'x'] = df_validation[building_indexes_validation]['LATITUDE'] - means[0]
                df_validation.loc[building_indexes_validation,'y'] = df_validation[building_indexes_validation]['LONGITUDE'] - means[1]

        # drop columns without training information
        mask = df_training[input_columns] != 0
        number_of_samples = np.sum(mask, axis=0)
        columns_without_information_names = list(number_of_samples.index[number_of_samples == 0])
        df_training = df_training.drop(columns=columns_without_information_names)
        df_validation = df_validation.drop(columns=columns_without_information_names)

        input_columns = list(df_training.filter(regex=r'WAP\d+').columns)

        return df_training, df_validation, input_columns

    def drop_rows_without_information(self, df, input_columns):
        mask = df[input_columns] != 100
        number_of_aps = np.sum(mask, axis=1)
        indexes_without_information = number_of_aps.index[number_of_aps == 0]
        clean_df = df.drop(indexes_without_information).reset_index(drop=True)
        return clean_df

    def transform(self, df, input_columns):
        # Sanity check
        if any(df[input_columns[0]] == 100):
            df = df.replace(100, -105)
            df[input_columns] = df[input_columns] + 105
        return df.copy()

    def get_X(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.df_training[self.input_columns].to_numpy(), self.df_validation[self.input_columns].to_numpy()

    # TODO: deprecate in favor of getX
    def get_X_df(self):
        return self.df_training[self.input_columns], self.df_validation[self.input_columns]

    def get_full_df(self):
        return self.df_training

    def get_categorical_y(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays of Bulding and Floor information

        Returns:
            (NDArray, NDArray): numpy arrays with Bulding and Floor information for training and validation datasets
        """
        y_train = np.hstack([(self.df_training['BUILDINGID']).to_numpy().reshape(-1, 1),
                             (self.df_training['FLOOR']).to_numpy().reshape(-1, 1)])
        y_test = np.hstack([(self.df_validation['BUILDINGID']).to_numpy().reshape(-1, 1),
                            (self.df_validation['FLOOR']).to_numpy().reshape(-1, 1)])
        return y_train, y_test

    def get_floor_data(self, building=None, floor=None, phoneid=None, reset_means=True):
        df_train = self.df_training
        df_test = self.df_validation
        if building is not None:
            df_train = df_train[df_train['BUILDINGID'] == building]
            df_test = df_test[df_test['BUILDINGID'] == building]
            # building_means = calculate_means(df_train)
        if floor is not None:
            df_train = df_train[df_train['FLOOR'] == floor]
            df_test = df_test[df_test['FLOOR'] == floor]
        if phoneid is not None:
            df_train = df_train[df_train['PHONEID'] == phoneid]
            df_test = df_test[df_test['PHONEID'] == phoneid]

        return UJIIndoorLoc(df_train, df_test, transform=False)

    def get_normalized_y(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and validation DataFrames with x and y columns only

        Returns:
            -(DataFrame): output DataFrame

        """
        normalized_y_train = self._get_normalized_df(self.df_training)
        normalized_y_test = self._get_normalized_df(self.df_validation)
        return normalized_y_train, normalized_y_test

    def get_continuous_y(self) -> Tuple[np.ndarray, np.ndarray]:
        continuous_columns = ['LATITUDE', 'LONGITUDE']
        return self.df_training[continuous_columns].to_numpy(), self.df_validation[continuous_columns].to_numpy()

    def _get_normalized_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get DataFrame with x and y columns

        Args:
            -df (DataFrame): input DataTrame

        Returns:
            -df (DataFrame): output DataFrame

        """
        return df.loc[:, ['x','y']]

    def compute_dimensions(self, points: NDArray):
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        polygon = Polygon(hull_points)
        obb = polygon.minimum_rotated_rectangle
        obb_coords = np.array(obb.exterior.coords)

        edges_obb = np.linalg.norm(obb_coords[1:] - obb_coords[:-1], axis=1)
        length_obb, width_obb = np.sort(edges_obb[:2])[::-1]

        plt.figure(figsize=(8, 8))
        plt.plot(points[:, 0], points[:, 1], 'o', label='Original Points')

        hull_closed = np.vstack([hull_points, hull_points[0]])
        plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'k--', label='Convex Hull')

        obb_closed = np.vstack([obb_coords, obb_coords[0]])
        plt.plot(obb_closed[:, 0], obb_closed[:, 1], 'r-', linewidth=2, label='OBB (Min Rotated Box)')

        plt.title(f"Polygon with OBB\nOBB - Length: {length_obb:.2f} m, Width: {width_obb:.2f} m")
        plt.xlabel("UTM X (meters)")
        plt.ylabel("UTM Y (meters)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.savefig("figures/points_dimensions")
        plt.show()

        return length_obb, width_obb
