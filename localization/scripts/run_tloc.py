import pickle
import argparse
import time
import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from scipy.interpolate import Rbf

from localization import utils, dataset


def filter_matrix_by_row(y, target_y):
    """
    :param y: matrix N x O
    :param target_y: vector 1 x O
    :return: indexes where y matches target_y
    """
    match = (y == target_y)
    row_equal = np.all(match, axis=1)
    indexes = np.where(row_equal)
    return indexes


def evaluate_model(continuous_location_model, discrete_model, dataset, y_real):
    start_time = time.time()
    _, X_test = dataset.get_X()
    predicted_discrete_location = discrete_model.predict(X_test)
    predicted_continuous_location = np.zeros((X_test.shape[0], 4))

    _, X_test = dataset.get_X_df()
    locations, counts = np.unique(predicted_discrete_location, axis=0, return_counts=True)
    # for each discrete location, apply the respective model
    for location in locations:
        indexes = filter_matrix_by_row(predicted_discrete_location, location)
        x_test_in_loc = X_test.iloc[indexes]
        y_pred, dist = continuous_location_model.pred(x_test_in_loc, selected_building=location[0],
                                                      selected_floor=location[1])
        predicted_continuous_location[indexes, :] = y_pred

    y_pred = np.vstack([predicted_continuous_location[:, 2], predicted_continuous_location[:, 3]]).T
    end_time = time.time()
    distances = np.linalg.norm(y_pred - y_real, axis=1)
    elapsed_seconds = (end_time - start_time)
    return distances, elapsed_seconds


def train_discrete_location_model(x_train, y, metric, k):
    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
    classifier = MultiOutputClassifier(neigh, n_jobs=-1)
    classifier.fit(x_train, y)
    return classifier


def evaluate_categorical_location_model(model, x_test, y_test):
    start_time = time.time()
    y_hat = model.predict(x_test)
    end_time = time.time()

    correct_indexes = np.argwhere(((y_hat == y_test).sum(axis=1) == 2)).flatten()
    score = correct_indexes.shape[0] / y_hat.shape[0]

    elapsed_seconds = (end_time - start_time)
    return score, correct_indexes, elapsed_seconds

def error_plot(data, name, b, f):
    import matplotlib.pyplot as plt
    # try:
    #     import scienceplots
    #     plt.style.use(['science','ieee'])
    # except:
    #     print("SciencePlots not found")

    scores = data
    threshold = np.percentile(distances, 90)

    plt.figure()
    plt.scatter(range(len(scores)),scores,c=~np.array([scores <= threshold]),cmap="bwr", s=2)
    plt.plot(range(len(scores)), threshold*np.ones(len(scores)),'g',linewidth=2)
    plt.title("Model outliers")
    plt.xlabel('Index')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    # plt.legend()
    plt.savefig(f"figures/outliers_{name}_b{b}f{f}")
    # plt.show()

def radial_log_basis_function(self, r):
    return np.log(r + self.epsilon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tLoc')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-b','--building', type=int, default=None, help='Building')
    parser.add_argument('-f', '--floor', type=int, default=None, help='Random seed')
    parser.add_argument('-m', '--model_name', type=str, default="model_gen_100", help='specification of the model')
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    utils.make_deterministic(args.seed)

    Rbf.radial_log_basis_function = radial_log_basis_function
    with open(f'output/{args.model_name}.bin', 'rb') as inp:
        model = pickle.load(inp)

    dataset = dataset.load_ujiindoor_loc(data_folder='data/filtered')
    dataset = dataset.get_floor_data(building=args.building, floor=args.floor, reset_means=True)

    x_train, X_test = dataset.get_X()
    y_train, y_test = dataset.get_categorical_y()
    y_continuous_train, y_continuous_test = dataset.get_normalized_y()

    discrete_location_model = train_discrete_location_model(x_train, y_train, 'euclidean', 3)
    score, correct_indexes, elapsed_time = evaluate_categorical_location_model(discrete_location_model, X_test, y_test)
    file_name = f"output/{args.model_name}/results.csv"
    if not os.path.exists(file_name):
        with open(file_name, "w") as out_file:
            out_file.write("accuracy,time,mean_error,median_error,P90,P95,mean_error_wod,median_error_wod,P90_wod,P95_wod,building,floor\n")
    print('Building and floor accuracy:', np.round(100 * score, 2))
    print('Prediction time:', np.round(elapsed_time, 2), 's\n')
    results = f"{np.round(100 * score, 2)},{np.round(elapsed_time, 2)},"

    distances, elapsed_time = evaluate_model(model, discrete_location_model, dataset, y_continuous_test)
    print(f'Mean error = {np.mean(distances):.2f}, median error = {np.median(distances):.2f}, '
          f'P90 = {np.percentile(distances, 90):.2f}, P95 = {np.percentile(distances, 95):.2f}')
    print('Removing the discrete position errors:')
    results += f"{np.mean(distances):.2f},{np.median(distances):.2f},{np.percentile(distances, 90):.2f},{np.percentile(distances, 95):.2f},"

    b = "all" if args.building is None else args.building
    f = "all" if args.floor is None else args.floor
    error_plot(distances, args.model_name, b, f)

    distances = distances[correct_indexes]
    print(f'Mean error = {np.mean(distances):.2f}, median error = {np.median(distances):.2f}, '
          f'P90 = {np.percentile(distances, 90):.2f}, P95 = {np.percentile(distances, 95):.2f}')
    print('Prediction time:', np.round(elapsed_time, 2), 's')
    results += f"{np.mean(distances):.2f},{np.median(distances):.2f},{np.percentile(distances, 90):.2f},{np.percentile(distances, 95):.2f},{b},{f}\n"
    print('End of execution')
    with open(file_name, "a") as out_file:
        out_file.write(results)
