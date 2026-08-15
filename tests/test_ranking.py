"""Tests for the feature_ranking orchestrator with the univariate methods."""

import numpy as np
import pandas as pd
import pytest

from featureranker.ranking import feature_ranking
from featureranker.univariate import MutualInfoOptions

UNIVARIATE = ["mi", "f_test"]
INFORMATIVE = {"feat_0", "feat_1", "feat_2", "feat_3"}


def test_invalid_task(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="task"):
        feature_ranking(X, y, task="ranking", methods=UNIVARIATE)


def test_invalid_dtype(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="dtype"):
        feature_ranking(X, y, methods=UNIVARIATE, dtype="float16")


def test_empty_methods(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="methods is empty"):
        feature_ranking(X, y, methods=[])


def test_duplicate_methods(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="more than once"):
        feature_ranking(X, y, methods=["mi", "mi"])


def test_unknown_method(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="Unknown methods"):
        feature_ranking(X, y, methods=["mi", "pca"])


def test_stray_options(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="will not run"):
        feature_ranking(X, y, methods=["mi"], options={"f_test": {}})


def test_unknown_option_key(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="Unknown options"):
        feature_ranking(X, y, methods=["mi"], options={"mi": {"neighbors": 5}})


def test_bad_n_jobs(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="n_jobs"):
        feature_ranking(X, y, methods=UNIVARIATE, n_jobs=0)


def test_rejects_numpy_features(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(TypeError, match="DataFrame"):
        feature_ranking(X.to_numpy(), y, methods=UNIVARIATE)


def test_duplicate_columns(synthetic_classification):
    X, y = synthetic_classification
    X = X.copy()
    X.columns = ["dup"] * X.shape[1]
    with pytest.raises(ValueError, match="duplicate"):
        feature_ranking(X, y, methods=UNIVARIATE)


def test_nan_in_features(synthetic_classification):
    X, y = synthetic_classification
    X = X.copy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        feature_ranking(X, y, methods=UNIVARIATE)


def test_non_numeric_features(synthetic_classification):
    X, y = synthetic_classification
    X = X.assign(kind="text")
    with pytest.raises(ValueError, match="non-numeric"):
        feature_ranking(X, y, methods=UNIVARIATE)


def test_length_mismatch(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="rows"):
        feature_ranking(X, y.iloc[:-5], methods=UNIVARIATE)


def test_nan_in_target(synthetic_classification):
    X, y = synthetic_classification
    y = y.astype(float).copy()
    y.iloc[3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        feature_ranking(X, y, methods=UNIVARIATE)


def test_single_class_target(synthetic_classification):
    X, y = synthetic_classification
    with pytest.raises(ValueError, match="2 classes"):
        feature_ranking(X, pd.Series([1] * len(y)), methods=UNIVARIATE)


def test_constant_regression_target(synthetic_regression):
    X, _, _ = synthetic_regression
    with pytest.raises(ValueError, match="constant"):
        feature_ranking(
            X, pd.Series([2.5] * len(X)), task="regression", methods=UNIVARIATE
        )


def test_classification_end_to_end(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=UNIVARIATE)
    assert result.task == "classification"
    assert result.methods == ("mi", "f_test")
    assert result.n_samples == 300
    assert result.n_features == 12
    assert result.classes == (0, 1)
    for method in UNIVARIATE:
        table = result.rankings[method]
        assert list(table.columns) == ["feature", "score"]
        assert set(table["feature"]) == set(X.columns)
        top4 = set(table["feature"].head(4))
        assert top4 == INFORMATIVE, f"{method} missed informative features: {top4}"


def test_regression_end_to_end(synthetic_regression):
    X, y, _ = synthetic_regression
    result = feature_ranking(X, y, task="regression", methods=UNIVARIATE)
    assert result.classes is None
    informative = {"feat_0", "feat_1", "feat_2"}
    for method in UNIVARIATE:
        top3 = set(result.rankings[method]["feature"].head(3))
        assert len(top3 & informative) >= 2, f"{method} missed informative features"


def test_string_target_encoded(synthetic_classification):
    X, y = synthetic_classification
    y_named = y.map({0: "healthy", 1: "sick"})
    result = feature_ranking(X, y_named, methods=UNIVARIATE)
    assert result.classes == ("healthy", "sick")


def test_mi_subsample_is_deterministic(synthetic_classification):
    X, y = synthetic_classification
    kwargs = dict(methods=["mi"], options={"mi": {"max_samples": 50}})
    first = feature_ranking(X, y, **kwargs)
    second = feature_ranking(X, y, **kwargs)
    assert first.diagnostics["mi"]["subsampled"] is True
    assert first.diagnostics["mi"]["n_samples_used"] == 50
    assert first.rankings["mi"].equals(second.rankings["mi"])


def test_options_accept_dataclass_instance(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(
        X, y, methods=["mi"], options={"mi": MutualInfoOptions(max_samples=None)}
    )
    assert result.diagnostics["mi"]["subsampled"] is False
