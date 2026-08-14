"""Whole-pipeline determinism guarantees."""

import pandas as pd

from featureranker.ranking import feature_ranking
from featureranker.vote import voting


def test_full_classification_two_runs_identical(synthetic_classification):
    X, y = synthetic_classification
    first = feature_ranking(X, y)
    second = feature_ranking(X, y)
    assert first.methods == ("rf", "xg", "mi", "f_test", "l1")
    assert first.equals(second)
    pd.testing.assert_frame_equal(voting(first), voting(second))


def test_regression_two_runs_identical(synthetic_regression):
    X, y, _ = synthetic_regression
    methods = ["mi", "f_test", "l1"]
    first = feature_ranking(X, y, task="regression", methods=methods)
    second = feature_ranking(X, y, task="regression", methods=methods)
    assert first.equals(second)


def test_parallel_matches_sequential(synthetic_classification):
    X, y = synthetic_classification
    methods = ["l1", "mi"]
    serial = feature_ranking(X, y, methods=methods, n_jobs=1)
    parallel = feature_ranking(X, y, methods=methods, n_jobs=4)
    assert serial.equals(parallel)
