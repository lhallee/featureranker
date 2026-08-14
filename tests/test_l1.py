"""Tests for the L1 rankers: lasso entry alphas and logistic entry Cs."""

import numpy as np
import pandas as pd
import pytest

from featureranker.lasso import LassoOptions, _choose_strategy
from featureranker.ranking import feature_ranking

REGRESSION_INFORMATIVE = {"feat_0", "feat_1", "feat_2"}


def _l1_scores(result) -> pd.Series:
    return result.rankings["l1"].set_index("feature")["score"]


@pytest.fixture(scope="module")
def wide_regression():
    """50 x 200 regression frame (p > n); feat_0..feat_4 informative."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 200))  # (n, p)
    w = np.array([4.0, -3.0, 5.0, -4.0, 3.0])  # (5,)
    y = X[:, :5] @ w + rng.normal(scale=0.5, size=50)  # (n,)
    frame = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(200)])
    return frame, pd.Series(y, name="target")


def test_auto_strategy_selection():
    options = LassoOptions()
    assert _choose_strategy(300, 12, options) == "exact"
    assert _choose_strategy(2_000, 5_000, options) == "grid"
    assert _choose_strategy(10_000_000, 1_000, options) == "grid"
    assert _choose_strategy(10_000_000, 1_000, LassoOptions(strategy="exact")) == "exact"


def test_lasso_exact_recovers_informative(synthetic_regression):
    X, y, _ = synthetic_regression
    result = feature_ranking(X, y, task="regression", methods=["l1"])
    assert result.diagnostics["l1"]["strategy"] == "exact"
    top3 = set(result.rankings["l1"]["feature"].head(3))
    assert top3 == REGRESSION_INFORMATIVE


def test_lasso_grid_recovers_informative(synthetic_regression):
    X, y, _ = synthetic_regression
    result = feature_ranking(
        X, y, task="regression", methods=["l1"],
        options={"l1": {"strategy": "grid"}},
    )
    assert result.diagnostics["l1"]["strategy"] == "grid"
    top3 = set(result.rankings["l1"]["feature"].head(3))
    assert top3 == REGRESSION_INFORMATIVE


def test_lasso_exact_and_grid_agree(synthetic_regression):
    """Exact knots must lie within one log-step of the grid's bracket estimates.

    Rank correlation is the wrong gate at small p: noise features with nearly
    identical entry alphas legitimately swap order between routes. The grid
    detects shallow crossings a few steps late even at tol=1e-6, so the gate
    is 10 grid steps on the log scale.
    """
    X, y, _ = synthetic_regression
    n_alphas, eps = 1000, 1e-4
    exact = feature_ranking(
        X, y, task="regression", methods=["l1"],
        options={"l1": {"strategy": "exact"}},
    )
    grid = feature_ranking(
        X, y, task="regression", methods=["l1"],
        options={"l1": {"strategy": "grid", "n_alphas": n_alphas, "eps": eps}},
    )
    e = _l1_scores(exact)
    g = _l1_scores(grid).reindex(e.index)

    entered_both = (e > 0) & (g > 0)
    log_step = np.log(1.0 / eps) / (n_alphas - 1)
    log_gap = np.abs(np.log(e[entered_both]) - np.log(g[entered_both]))
    assert log_gap.max() <= 10 * log_step

    # entries the grid missed must sit below its floor of alpha_max * eps
    grid_missed = (e > 0) & (g == 0)
    assert (e[grid_missed] <= e.max() * eps * 2).all()
    assert not ((e == 0) & (g > 0)).any()


def test_lasso_wide_data(wide_regression):
    X, y = wide_regression
    result = feature_ranking(X, y, task="regression", methods=["l1"])
    scores = _l1_scores(result)
    top5 = set(result.rankings["l1"]["feature"].head(5))
    assert top5 == {"feat_0", "feat_1", "feat_2", "feat_3", "feat_4"}
    # p > n: the path can activate at most ~n features, the rest score zero
    assert result.diagnostics["l1"]["n_never_entered"] >= 100
    assert (scores == 0.0).sum() == result.diagnostics["l1"]["n_never_entered"]


def test_lasso_wide_grid_matches_top_features(wide_regression):
    X, y = wide_regression
    result = feature_ranking(
        X, y, task="regression", methods=["l1"],
        options={"l1": {"strategy": "grid"}},
    )
    top5 = set(result.rankings["l1"]["feature"].head(5))
    assert top5 == {"feat_0", "feat_1", "feat_2", "feat_3", "feat_4"}


def test_lasso_degeneracy_falls_back_to_grid(synthetic_regression, monkeypatch):
    X, y, _ = synthetic_regression

    def broken_lars(*args, **kwargs):
        raise np.linalg.LinAlgError("forced degeneracy")

    monkeypatch.setattr("featureranker.lasso.lars_path_gram", broken_lars)
    result = feature_ranking(X, y, task="regression", methods=["l1"])
    assert result.diagnostics["l1"]["strategy"] == "grid"
    top3 = set(result.rankings["l1"]["feature"].head(3))
    assert top3 == REGRESSION_INFORMATIVE


def test_lasso_deterministic(synthetic_regression):
    X, y, _ = synthetic_regression
    first = feature_ranking(X, y, task="regression", methods=["l1"])
    second = feature_ranking(X, y, task="regression", methods=["l1"])
    assert first.equals(second)


CLASSIFICATION_INFORMATIVE = {"feat_0", "feat_1", "feat_2", "feat_3"}


def test_logistic_recovers_informative(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["l1"])
    diagnostics = result.diagnostics["l1"]
    assert diagnostics["solver"] == "liblinear"
    assert diagnostics["n_fits"] <= 128
    top4 = set(result.rankings["l1"]["feature"].head(4))
    assert top4 == CLASSIFICATION_INFORMATIVE


def test_logistic_entry_direction(synthetic_classification):
    """Informative features must enter at smaller C than noise features.

    Guards the fixed v2 bug that recorded the largest C instead of the entry.
    """
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["l1"])
    entry_C = result.diagnostics["l1"]["entry_C"]  # (p,) in feature order
    assert np.isfinite(entry_C[:4]).all()
    assert entry_C[:4].mean() < entry_C[4:].mean()


def test_logistic_multiclass(synthetic_multiclass):
    """v2 crashed on 3+ classes; the any-class entry mask must handle them."""
    X, y = synthetic_multiclass
    result = feature_ranking(X, y, methods=["l1"])
    assert result.diagnostics["l1"]["solver"] == "saga"
    assert set(result.rankings["l1"]["feature"]) == set(X.columns)
    top4 = set(result.rankings["l1"]["feature"].head(4))
    assert top4 == CLASSIFICATION_INFORMATIVE


def test_logistic_forced_saga_binary(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(
        X, y, methods=["l1"], options={"l1": {"solver": "saga"}}
    )
    assert result.diagnostics["l1"]["solver"] == "saga"
    top4 = set(result.rankings["l1"]["feature"].head(4))
    assert top4 == CLASSIFICATION_INFORMATIVE


def test_logistic_liblinear_multiclass_raises(synthetic_multiclass):
    X, y = synthetic_multiclass
    with pytest.raises(ValueError, match="binary"):
        feature_ranking(X, y, methods=["l1"], options={"l1": {"solver": "liblinear"}})


def test_logistic_wave_matches_dense_oracle(synthetic_classification):
    """Wave refinement must land within rtol of a dense one-wave grid."""
    X, y = synthetic_classification
    dense_points, dense_decades = 400, 5.0
    waves = feature_ranking(X, y, methods=["l1"])
    dense = feature_ranking(
        X, y, methods=["l1"],
        options={"l1": {
            "coarse_size": dense_points, "decades": dense_decades,
            "max_waves": 1, "max_fits": dense_points,
        }},
    )
    entry_waves = waves.diagnostics["l1"]["entry_C"]  # (p,)
    entry_dense = dense.diagnostics["l1"]["entry_C"]  # (p,)

    both = np.isfinite(entry_waves) & np.isfinite(entry_dense)
    dense_step = np.log(10.0**dense_decades) / (dense_points - 1)
    gate = np.log(1.15) + 2 * dense_step
    log_gap = np.abs(np.log(entry_waves[both]) - np.log(entry_dense[both]))
    assert log_gap.max() <= gate


def test_logistic_wide_smoke(wide_data):
    X, y = wide_data
    result = feature_ranking(X, y, methods=["l1"])
    assert set(result.rankings["l1"]["feature"]) == set(X.columns)
    top10 = set(result.rankings["l1"]["feature"].head(10))
    informative = {f"feat_{i}" for i in range(10)}
    assert len(top10 & informative) >= 8


def test_logistic_deterministic(synthetic_classification):
    X, y = synthetic_classification
    first = feature_ranking(X, y, methods=["l1"])
    second = feature_ranking(X, y, methods=["l1"])
    assert first.equals(second)
