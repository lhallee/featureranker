"""Tests for the simplex-constrained convex combination fit."""

import numpy as np
import pandas as pd
import pytest

from featureranker import feature_ranking, fit_convex


@pytest.fixture(scope="module")
def planted_regression():
    """y is exactly 0.7 * a + 0.3 * b; c and d are pure noise."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": rng.normal(size=n),
    })
    y = pd.Series(0.7 * X["a"] + 0.3 * X["b"], name="target")
    return X, y


def test_fit_convex_raw_recovers_planted_weights(planted_regression):
    """standardize=False fits the raw features, so recovery is exact."""
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression", standardize=False)
    weight_of = dict(zip(fit.feature_names, fit.weights))
    assert weight_of["a"] == pytest.approx(0.7, abs=0.02)
    assert weight_of["b"] == pytest.approx(0.3, abs=0.02)
    assert fit.metric_name == "r2"
    assert fit.metric_value > 0.99
    assert fit.feature_means is None


def test_fit_convex_standardize_handles_mixed_scales(planted_regression):
    """The default z-scoring keeps huge-scale columns from drowning the fit."""
    X, y = planted_regression
    rescaled = X.assign(a=X["a"] * 10_000.0)
    fit = fit_convex(rescaled, y, task="regression")
    weight_of = dict(zip(fit.feature_names, fit.weights))
    assert weight_of["a"] == pytest.approx(0.7, abs=0.05)
    assert weight_of["b"] == pytest.approx(0.3, abs=0.05)
    assert fit.metric_value > 0.98


def test_fit_convex_weights_on_simplex(planted_regression):
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    assert (fit.weights >= 0.0).all()
    assert fit.weights.sum() == pytest.approx(1.0)


def test_fit_convex_is_deterministic(planted_regression):
    X, y = planted_regression
    first = fit_convex(X, y, task="regression")
    second = fit_convex(X, y, task="regression")
    assert np.array_equal(first.weights, second.weights)


def test_fit_convex_binary_classification(planted_regression):
    X, _ = planted_regression
    y = (0.8 * X["a"] + 0.2 * X["b"] > 0).astype(int)
    fit = fit_convex(X, y, task="classification")
    assert fit.metric_name == "auc"
    assert fit.metric_value > 0.95
    assert fit.weights.sum() == pytest.approx(1.0)


def test_fit_convex_rejects_multiclass(planted_regression):
    X, _ = planted_regression
    y = np.arange(len(X)) % 3
    with pytest.raises(ValueError, match="binary"):
        fit_convex(X, y, task="classification")


def test_fit_convex_bad_task(planted_regression):
    X, y = planted_regression
    with pytest.raises(ValueError, match="task"):
        fit_convex(X, y, task="ranking")


def test_predict_raw_matches_manual_score(planted_regression):
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression", standardize=False)
    scores = fit.predict(X)
    expected = X[list(fit.feature_names)].to_numpy() @ fit.weights
    assert np.allclose(scores, expected)


def test_predict_standardized_reproduces_fit_metric(planted_regression):
    """predict must apply the stored transform: its R2 equals the fit metric."""
    from sklearn.metrics import r2_score

    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    assert r2_score(y, fit.predict(X)) == pytest.approx(fit.metric_value)


def test_predict_dataframe_ignores_extra_columns(planted_regression):
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    widened = X.assign(extra=1.0)
    assert np.allclose(fit.predict(widened), fit.predict(X))


def test_predict_missing_feature_raises(planted_regression):
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    with pytest.raises(ValueError, match="missing"):
        fit.predict(X.drop(columns=["a"]))


def test_predict_numpy_wrong_width_raises(planted_regression):
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    with pytest.raises(ValueError, match="columns"):
        fit.predict(np.zeros((5, 2)))


def test_table_sorted_by_weight(planted_regression):
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    table = fit.table()
    assert list(table.columns) == ["feature", "weight"]
    assert list(table["feature"].head(2)) == ["a", "b"]
    assert table["weight"].is_monotonic_decreasing


def test_result_fit_convex_top_n(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    fit = result.fit_convex(X, y, top_n=2, standardize=False)
    assert set(fit.feature_names) == {"a", "b"}
    weight_of = dict(zip(fit.feature_names, fit.weights))
    assert weight_of["a"] == pytest.approx(0.7, abs=0.02)
    assert fit.metric_value > 0.99


def test_result_fit_convex_default_standardizes(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    fit = result.fit_convex(X, y, top_n=2)
    assert fit.feature_means is not None
    assert fit.weights.sum() == pytest.approx(1.0)
    assert fit.metric_value > 0.98


def test_result_fit_convex_defaults_to_all_features(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    fit = result.fit_convex(X, y)
    assert len(fit.feature_names) == X.shape[1]


def test_result_fit_convex_top_n_out_of_range(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    with pytest.raises(ValueError, match="top_n"):
        result.fit_convex(X, y, top_n=0)
    with pytest.raises(ValueError, match="top_n"):
        result.fit_convex(X, y, top_n=99)


def test_result_fit_convex_mismatched_columns(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    renamed = X.rename(columns={"a": "z"})
    with pytest.raises(ValueError, match="match"):
        result.fit_convex(renamed, y)
