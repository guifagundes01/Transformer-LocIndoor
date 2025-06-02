import pickle
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

from localization.clusterization import build_distance_matrix, show_clusters_on_map

def radial_log_basis_function(self, r):
    return np.log(r + self.epsilon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Clusterization')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-df', '--data_folder', type=str, default="data/generated", help='Folder to save the generated data')
    parser.add_argument('-s', '--size', type=int, default=100, help='Value of discretization of the model')
    parser.add_argument('-fr', '--format', type=str, default="eps", help='Image format to save')
    parser.add_argument('-b','--building', type=int, default=None, help='Building number')
    parser.add_argument('-f','--floor', type=int, default=None, help='Floor number')
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    with open(f"{args.data_folder}/power_distribution_building_{args.building}_floor_{args.floor}.json") as f:
        data = json.load(f)

    dist_matrix = build_distance_matrix(data)

    np.save(f"{args.data_folder}/dist_matrix_b{args.building}f{args.floor}_{args.size}.npy", dist_matrix)
    matrix = dist_matrix
    np.fill_diagonal(matrix, 0)

    thresholds = np.linspace(0.01, 1.0, 100)
    scores = []
    n_clusters_list = []
    best_score = -1
    best_thresh = None

    for t in thresholds:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=t,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(dist_matrix)
            n_clusters = len(set(labels))
            if n_clusters <= 1 or n_clusters >= len(dist_matrix) - 1:
                scores.append(np.nan)
            else:
                score = silhouette_score(matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_thresh = t
                scores.append(score)
            n_clusters_list.append(n_clusters)
        except Exception as e:
            print(f"Threshold {t:.2f}: Error - {e}")
            scores.append(np.nan)
            n_clusters_list.append(0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(thresholds, scores, marker='o', label='Silhouette Score', color='#1f77b4')
    ax1.set_xlabel('Distance Threshold')
    ax1.set_ylabel('Silhouette Score', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    # ax1.set_title('Silhouette Score vs. Agglomerative Clustering Threshold')

    ax2 = ax1.twinx()
    ax2.plot(thresholds, n_clusters_list, marker='s', label='Number of Clusters', color='#2ca02c')
    ax2.set_ylabel('Number of Clusters', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')

    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f"silhoette.{args.format}", format=args.format)

    Rbf.radial_log_basis_function = radial_log_basis_function
    with open(f'output/filtered_model_{args.size}.bin', 'rb') as inp:
        model = pickle.load(inp)

    x = model.x_building[0]
    y = model.y_building[0]

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=best_thresh,
        n_clusters=None
    )
    labels = clustering.fit_predict(dist_matrix)
    len(np.unique(labels))

    show_clusters_on_map(x, y, labels, point_ids=list(range(dist_matrix.shape[0])), format=args.format)
