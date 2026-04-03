import numpy as np
import pytest

from featureranker.clustering import (
    get_inertia,
    get_kmean_metrics,
    optimal_k_w_both,
    optimal_k_w_elbow,
    random_cluster_generator,
)


def test_random_cluster_generator():
    X = random_cluster_generator(n_samples=100, n_features=3, n_centers=4)
    assert X.shape == (100, 3)


def test_get_inertia():
    X = random_cluster_generator(n_samples=200, n_centers=3)
    inertia = get_inertia(X, 3)
    assert inertia > 0


def test_inertia_monotonically_decreases():
    X = random_cluster_generator(n_samples=200, n_centers=3)
    inertias = [get_inertia(X, k) for k in range(1, 6)]
    for i in range(len(inertias) - 1):
        assert inertias[i] >= inertias[i + 1]


def test_optimal_k_w_elbow():
    X = random_cluster_generator(n_samples=300, n_centers=3, std=0.5)
    k = optimal_k_w_elbow(X, max_k=8)
    assert 2 <= k <= 8


def test_optimal_k_w_elbow_assert():
    X = random_cluster_generator()
    with pytest.raises(AssertionError):
        optimal_k_w_elbow(X, max_k=1)


def test_get_kmean_metrics():
    X = random_cluster_generator(n_samples=100, n_centers=3)
    inertia, sil = get_kmean_metrics(X, 3)
    assert inertia > 0
    assert -1 <= sil <= 1


def test_get_kmean_metrics_k1():
    X = random_cluster_generator(n_samples=100, n_centers=3)
    inertia, sil = get_kmean_metrics(X, 1)
    assert inertia > 0
    assert sil == 0.0


def test_optimal_k_w_both():
    X = random_cluster_generator(n_samples=300, n_centers=3, std=0.5)
    k = optimal_k_w_both(X, max_k=8)
    assert 2 <= k <= 8
