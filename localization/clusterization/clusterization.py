import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import AgglomerativeClustering

import ot

def normalize(dist):
    arr = np.array(dist, dtype=np.float64)
    total = arr.sum()
    return arr / total if total > 0 else arr

def js_similarity(p1, p2):
    return 1 - jensenshannon(normalize(p1), normalize(p2))

def js_distance(p1, p2):
    return jensenshannon(normalize(p1), normalize(p2))

def average_router_similarity(pointA, pointB):
    shared = set(pointA.keys()) & set(pointB.keys())
    if not shared:
        return 0.0 # None
    similarities = [
        js_similarity(pointA[r], pointB[r])
        for r in shared
    ]
    return sum(similarities) / len(similarities)

def average_router_distance(pointA, pointB):
    shared = set(pointA.keys()) & set(pointB.keys())
    if not shared:
        return 1.0
    distances = [
        js_distance(pointA[r], pointB[r])
        for r in shared
    ]
    return sum(distances) / len(distances)

def emd_similarity(pointA, pointB):
    A_dists = [normalize(d) for d in pointA.values()]
    B_dists = [normalize(d) for d in pointB.values()]
    a = np.ones(len(A_dists)) / len(A_dists)
    b = np.ones(len(B_dists)) / len(B_dists)

    cost = np.zeros((len(A_dists), len(B_dists)))
    for i, distA in enumerate(A_dists):
        for j, distB in enumerate(B_dists):
            cost[i, j] = jensenshannon(distA, distB)

    emd_val = ot.emd2(a, b, cost)
    return 1 - emd_val

def compute_point_similarity(p1, p2, use_emd=True):
    avg_sim = average_router_similarity(p1, p2)
    if avg_sim is not None:
        return avg_sim
    elif use_emd:
        return emd_similarity(p1, p2)
    else:
        return 0.0

def build_similarity_matrix(data, use_emd=True):
    N = len(data)
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            sim = compute_point_similarity(data[i], data[j], use_emd)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix

def build_distance_matrix(data):
    N = len(data)
    dist_matrix = np.ones((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist = average_router_distance(data[i], data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def cluster_points(data, similarity_threshold=0.9, use_emd=True):
    # dist_matrix = build_similarity_matrix(data, use_emd)
    dist_matrix = build_distance_matrix(data)
    dist_matrix[np.argwhere(np.isnan(dist_matrix))] = 1.0

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(dist_matrix)
    return labels, dist_matrix

def show_clusters_on_map(x, y, labels, point_ids=None, title="Clustered Points on Spatial Map", format="eps"):
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=labels, cmap='nipy_spectral', s=10) # nipy_spectral, hsv
    # num_clusters = len(np.unique(labels))
    # for lbl in unique_labels:
    #     mask = labels == lbl
    #     if np.sum(mask) > 0:
    #         cx = x[mask].mean()
    #         cy = y[mask].mean()
    #         radius = 3

    #         circle = Circle((cx, cy), radius=radius, edgecolor='black',
    #                         facecolor='white', linewidth=1.5)
    #         plt.gca().add_patch(circle)

    #         plt.text(cx, cy, str(lbl), fontsize=10, fontweight='bold', color='black',
    #                 ha='center', va='center')

    # plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{title}.{format}", format=format)
