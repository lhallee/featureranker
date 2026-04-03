import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def random_cluster_generator(
    n_samples: int = 1000,
    n_features: int = 2,
    n_centers: int = 3,
    std: float = 1.0,
) -> np.ndarray:
    """Generate random clustered data using make_blobs."""
    return make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=std,
        random_state=42,
    )[0]


def get_inertia(X: np.ndarray, k: int) -> float:
    """Compute within-cluster sum of squares for a given k."""
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_


def optimal_k_w_elbow(X: np.ndarray, max_k: int = 10) -> int:
    """Find optimal k using the elbow method (max distance from baseline)."""
    assert max_k >= 2, f"max_k must be >= 2, got {max_k}"
    inertias = np.array([get_inertia(X, k) for k in range(1, max_k + 1)])
    slope = (inertias[-1] - inertias[0]) / (max_k - 1)
    linear = np.array([
        slope * x + (inertias[-1] - slope * max_k)
        for x in range(1, max_k + 1)
    ])
    return int((linear - inertias).argmax() + 1)


def get_kmean_metrics(X: np.ndarray, k: int) -> tuple[float, float]:
    """Return (inertia, silhouette_score) for a given k."""
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    if k < 2:
        return inertia, 0.0
    try:
        sil = silhouette_score(X, kmeans.labels_)
    except ValueError:
        sil = 0.0
    return inertia, sil


def optimal_k_w_both(X: np.ndarray, max_k: int = 10) -> int:
    """Find optimal k using combined elbow + silhouette scoring."""
    assert max_k >= 2, f"max_k must be >= 2, got {max_k}"
    metrics = [get_kmean_metrics(X, k) for k in range(1, max_k + 1)]
    inertias = np.array([m[0] for m in metrics])
    sils = np.array([m[1] for m in metrics])
    slope = (inertias[-1] - inertias[0]) / (max_k - 1)
    linear = np.array([
        slope * x + (inertias[-1] - slope * max_k)
        for x in range(1, max_k + 1)
    ])
    dists = linear - inertias
    scores = dists * sils
    return int(scores.argmax() + 1)
