"""Tests for the probe evaluation and probe-driven auto vote weights."""

import pytest

from featureranker import feature_ranking, voting


def test_probe_reports_attached(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["f_test", "mi"])
    for method in ("f_test", "mi"):
        report = result.diagnostics[method]["probe"]
        assert report["metric"] == "balanced_accuracy"
        assert set(report["by_k"]) == {1, 2, 4, 8}
        assert 0.0 <= report["skill"] <= 1.0


def test_probe_table(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["f_test", "mi"])
    table = result.probe_table()
    assert list(table.index) == ["f_test", "mi"]
    assert list(table.columns) == [1, 2, 4, 8, "score", "skill"]


def test_probe_regression_metric(synthetic_regression):
    X, y, _ = synthetic_regression
    result = feature_ranking(X, y, task="regression", methods=["f_test"])
    report = result.diagnostics["f_test"]["probe"]
    assert report["metric"] == "r2"
    assert report["skill"] > 0.0


def test_probe_disabled(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["f_test"], probe=False)
    assert "probe" not in result.diagnostics["f_test"]
    with pytest.raises(ValueError, match="probe"):
        result.probe_table()


def test_probe_is_deterministic(synthetic_classification):
    X, y = synthetic_classification
    first = feature_ranking(X, y, methods=["f_test"])
    second = feature_ranking(X, y, methods=["f_test"])
    assert first.diagnostics["f_test"]["probe"] == second.diagnostics["f_test"]["probe"]


def test_voting_auto_weights(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["f_test", "mi"])
    table = voting(result, weights="auto")
    assert len(table) == X.shape[1]
    assert set(table["feature"].head(4)) == {f"feat_{i}" for i in range(4)}


def test_voting_auto_requires_probe(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["f_test"], probe=False)
    with pytest.raises(ValueError, match="probe"):
        voting(result, weights="auto")


def test_voting_auto_rejects_plain_mapping(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["f_test"])
    with pytest.raises(ValueError, match="RankingResult"):
        voting(result.rankings, weights="auto")


def test_voting_rejects_unknown_string_weights(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y, methods=["f_test"])
    with pytest.raises(ValueError, match="auto"):
        voting(result, weights="automatic")
