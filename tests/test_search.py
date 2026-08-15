"""Tests for the tree-model halving search and the n_jobs budget mapper."""

import joblib
import numpy as np
import pandas as pd
import pytest

from featureranker.ranking import _resolve_budget, feature_ranking

INFORMATIVE = {"feat_0", "feat_1", "feat_2", "feat_3"}


def test_budget_mapper():
    cores = joblib.cpu_count()
    assert _resolve_budget(-1) == cores
    assert _resolve_budget(2) == min(2, cores)
    assert _resolve_budget(10_000) == cores
    with pytest.raises(ValueError, match="n_jobs"):
        _resolve_budget(0)
    with pytest.raises(ValueError, match="n_jobs"):
        _resolve_budget(-2)


def test_halving_tiny_dataset_falls_back():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = pd.Series(rng.integers(0, 2, size=30))
    result = feature_ranking(X, y, methods=["rf"], options={"rf": {"cv": 2}})
    assert result.diagnostics["rf"]["search"] == "randomized"
    assert set(result.rankings["rf"]["feature"]) == set("abcde")


def test_rf_recovers_informative(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["rf"])
    diagnostics = result.diagnostics["rf"]
    assert diagnostics["search"] == "halving"
    assert diagnostics["best_params"]
    top4 = set(result.rankings["rf"]["feature"].head(4))
    assert top4 == INFORMATIVE


def test_xgboost_recovers_informative(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["xg"])
    top4 = set(result.rankings["xg"]["feature"].head(4))
    assert top4 == INFORMATIVE


def test_rf_regression(synthetic_regression):
    X, y, _ = synthetic_regression
    result = feature_ranking(X, y, task="regression", methods=["rf"])
    top3 = set(result.rankings["rf"]["feature"].head(3))
    assert top3 == {"feat_0", "feat_1", "feat_2"}


def test_small_class_raises():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    y = pd.Series([0] * 38 + [1] * 2)
    with pytest.raises(ValueError, match="smallest class"):
        feature_ranking(X, y, methods=["rf"])


def test_estimator_kwargs_passthrough(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(
        X, y, methods=["rf"],
        options={"rf": {"estimator_kwargs": {"criterion": "entropy"}}},
    )
    assert set(result.rankings["rf"]["feature"]) == set(X.columns)


def test_rf_deterministic(synthetic_classification):
    X, y = synthetic_classification
    first = feature_ranking(X, y, methods=["rf"])
    second = feature_ranking(X, y, methods=["rf"])
    assert first.equals(second)
