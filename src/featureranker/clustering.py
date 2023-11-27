import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def random_cluster_generator(n_samples=1000, n_features=2, n_centers=3, std=1.0):
    return make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=std)[0]


def get_inertia(X, k):
    return KMeans(n_clusters=k).fit(X).inertia_


def optimal_k_w_elbow(X, max_k=10):
    inertias = np.array([get_inertia(X, k) for k in range(1, max_k+1)])
    slope = (inertias[max_k-1] - inertias[0]) / (max_k - 1)
    linear = np.array([slope * (x) + (inertias[max_k-1] - slope * max_k) for x in range(1, max_k+1)])
    return (linear-inertias).argmax(axis=0)+1


def get_kmean_metrics(X, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    try:
        silhouette = silhouette_score(X, kmeans.labels_)
    except:
        silhouette = 0
    return inertia, silhouette


def optimal_k_w_both(X, max_k=10):
    metrics = [get_kmean_metrics(X, k) for k in range(1, max_k+1)]
    inertias = np.array([metric[0] for metric in metrics])
    slope = (inertias[max_k-1] - inertias[0]) / (max_k - 1)
    linear = np.array([slope * (x) + (inertias[max_k-1] - slope * max_k) for x in range(1, max_k+1)])
    dists = linear - inertias
    sils = np.array([metric[1] for metric in metrics])
    scores = np.array([d * s for d, s in zip(dists, sils)])
    return scores.argmax()+1