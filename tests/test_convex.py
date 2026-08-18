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
    """standardize=False and entropy=0 fit the raw problem, so recovery is exact."""
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression", standardize=False, entropy=0.0)
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
    fit = fit_convex(rescaled, y, task="regression", entropy=0.0)
    weight_of = dict(zip(fit.feature_names, fit.weights))
    assert weight_of["a"] == pytest.approx(0.7, abs=0.05)
    assert weight_of["b"] == pytest.approx(0.3, abs=0.05)
    assert fit.metric_value > 0.98


def test_fit_convex_default_strictly_positive(planted_regression):
    """The default entropy smoothing keeps every weight strictly positive."""
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    assert (fit.weights > 0.0).all()
    assert fit.metric_value > 0.95
    weight_of = dict(zip(fit.feature_names, fit.weights))
    assert weight_of["a"] > weight_of["c"]
    assert weight_of["b"] > weight_of["d"]


def test_fit_convex_entropy_zero_allows_exact_zeros(planted_regression):
    """Plain least squares parks noise features exactly on the boundary."""
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression", standardize=False, entropy=0.0)
    assert (fit.weights == 0.0).any()


def test_fit_convex_entropy_resolves_duplicate_features():
    """Duplicated columns make plain least squares ill-posed; entropy picks
    the symmetric solution."""
    rng = np.random.default_rng(1)
    n = 300
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "a_copy": a, "noise": rng.normal(size=n)})
    y = pd.Series(a, name="target")
    fit = fit_convex(X, y, task="regression")
    weight_of = dict(zip(fit.feature_names, fit.weights))
    assert weight_of["a"] == pytest.approx(weight_of["a_copy"], abs=1e-6)
    assert weight_of["noise"] < weight_of["a"]


def test_fit_convex_negative_entropy_raises(planted_regression):
    X, y = planted_regression
    with pytest.raises(ValueError, match="entropy"):
        fit_convex(X, y, task="regression", entropy=-0.1)


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
    fit = result.fit_convex(X, y, top_n=2, standardize=False, entropy=0.0)
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


def test_result_fit_convex_top_n_clamps_to_feature_count(planted_regression):
    """A fixed NUM_KEEP above the feature count fits all features."""
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    fit = result.fit_convex(X, y, top_n=99)
    assert len(fit.feature_names) == X.shape[1]


def test_result_fit_convex_reports_method_metrics(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["mi", "f_test"])
    fit = result.fit_convex(X, y, top_n=2)
    assert list(fit.method_metrics) == ["mi", "f_test", "ensemble"]
    assert fit.method_metrics["ensemble"] == fit.metrics
    assert all("train" in values for values in fit.method_metrics.values())


def test_result_fit_convex_full_set_metrics_collapse(planted_regression):
    """With top_n covering all features every selection is the same set."""
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    fit = result.fit_convex(X, y)
    assert fit.method_metrics == {"f_test": fit.metrics, "ensemble": fit.metrics}


def test_standalone_fit_convex_has_no_method_metrics(planted_regression):
    X, y = planted_regression
    fit = fit_convex(X, y, task="regression")
    assert fit.method_metrics is None
    assert set(fit.metrics) == {"train"}
    assert fit.metric_value == fit.metrics["train"]


def _three_way_split(X, y):
    return (
        (X.iloc[:240], y.iloc[:240]),
        (X.iloc[240:320], y.iloc[240:320]),
        (X.iloc[320:], y.iloc[320:]),
    )


def test_fit_convex_reports_valid_and_test_metrics(planted_regression):
    X, y = planted_regression
    (X_tr, y_tr), valid, test = _three_way_split(X, y)
    fit = fit_convex(X_tr, y_tr, task="regression", valid=valid, test=test)
    assert list(fit.metrics) == ["train", "valid", "test"]
    assert fit.metrics["valid"] > 0.9
    assert fit.metrics["test"] > 0.9
    assert "test=" in repr(fit)


def test_result_fit_convex_eval_splits(planted_regression):
    X, y = planted_regression
    (X_tr, y_tr), valid, test = _three_way_split(X, y)
    result = feature_ranking(X_tr, y_tr, task="regression", methods=["f_test"])
    fit = result.fit_convex(X_tr, y_tr, top_n=2, valid=valid, test=test)
    for values in fit.method_metrics.values():
        assert set(values) == {"train", "valid", "test"}
    assert fit.method_metrics["ensemble"] == fit.metrics


def test_fit_convex_eval_pair_must_be_pair(planted_regression):
    X, y = planted_regression
    with pytest.raises(TypeError, match="pair"):
        fit_convex(X, y, task="regression", test=X)


def test_fit_convex_eval_unseen_label_raises(planted_regression):
    X, _ = planted_regression
    y = (X["a"] > 0).astype(int)
    y_test = y.iloc[:50].copy()
    y_test.iloc[0] = 2
    with pytest.raises(ValueError, match="never showed"):
        fit_convex(X, y, task="classification", test=(X.iloc[:50], y_test))


def test_result_fit_convex_eval_mismatched_features(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    renamed = X.rename(columns={"a": "z"})
    with pytest.raises(ValueError, match="match"):
        result.fit_convex(X, y, test=(renamed, y))


def test_result_fit_convex_mismatched_columns(planted_regression):
    X, y = planted_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    renamed = X.rename(columns={"a": "z"})
    with pytest.raises(ValueError, match="match"):
        result.fit_convex(renamed, y)
