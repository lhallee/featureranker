import numpy as np
import pandas as pd
import pytest

from featureranker.rankers import (
    feature_ranking,
    l1_classification_ranking,
    l1_regression_ranking,
    make_ranking,
    voting,
)


def test_make_ranking():
    cols = ["a", "b", "c"]
    scores = np.array([0.1, 0.5, 0.3])
    df = make_ranking("Test", cols, scores)
    assert list(df.columns) == ["Test", "Score"]
    assert df.iloc[0]["Test"] == "b"
    assert len(df) == 3


def test_make_ranking_length_mismatch():
    with pytest.raises(AssertionError, match="Length mismatch"):
        make_ranking("X", ["a", "b"], np.array([1.0]))


def test_l1_regression_ranking(diabetes_data):
    X, y = diabetes_data
    df = l1_regression_ranking(X, y)
    assert "L1" in df.columns
    assert "Score" in df.columns
    assert len(df) == X.shape[1]
    assert df["Score"].is_monotonic_decreasing


def test_l1_classification_ranking(cancer_data):
    X, y = cancer_data
    df = l1_classification_ranking(X, y, n_regularization_steps=10)
    assert "L1" in df.columns
    assert "Score" in df.columns
    assert len(df) == X.shape[1]


def test_feature_ranking_single_method(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi"])
    assert len(rankings) == 1
    name, df = rankings[0]
    assert name == "MI"
    assert len(df) == X.shape[1]


def test_feature_ranking_multiple_methods(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(
        X, y, task="classification", choices=["mi", "f_test"],
    )
    assert len(rankings) == 2
    names = [r[0] for r in rankings]
    assert "MI" in names
    assert "F" in names


def test_feature_ranking_invalid_choice(cancer_data):
    X, y = cancer_data
    with pytest.raises(AssertionError, match="Invalid choices"):
        feature_ranking(X, y, choices=["bad_method"])


def test_feature_ranking_invalid_task(cancer_data):
    X, y = cancer_data
    with pytest.raises(AssertionError, match="Invalid task"):
        feature_ranking(X, y, task="invalid")


def test_feature_ranking_regression(diabetes_data):
    X, y = diabetes_data
    rankings = feature_ranking(X, y, task="regression", choices=["mi", "f_test"])
    assert len(rankings) == 2


def test_voting_reciprocal_rank(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi", "f_test"])
    result = voting(rankings)
    assert "Feature" in result.columns
    assert "Score" in result.columns
    assert len(result) == X.shape[1]
    assert result["Score"].is_monotonic_decreasing


def test_voting_borda(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi", "f_test"])
    result = voting(rankings, method="borda")
    assert len(result) == X.shape[1]
    assert result["Score"].is_monotonic_decreasing


def test_voting_exponential(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi", "f_test"])
    result = voting(rankings, method="exponential")
    assert len(result) == X.shape[1]


def test_voting_with_weights(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi", "f_test"])
    result = voting(rankings, weights=[0.3, 0.7])
    assert len(result) == X.shape[1]


def test_voting_invalid_method(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi"])
    with pytest.raises(AssertionError, match="Invalid method"):
        voting(rankings, method="bad")


def test_voting_weight_mismatch(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi"])
    with pytest.raises(AssertionError, match="Length mismatch"):
        voting(rankings, weights=[1.0, 2.0])


def test_feature_ranking_save(cancer_data, tmp_path):
    X, y = cancer_data
    save_file = tmp_path / "rankings.pkl"
    rankings = feature_ranking(
        X, y, task="classification", choices=["mi"],
        save=True, save_path=str(save_file),
    )
    assert save_file.exists()


def test_voting_save(cancer_data, tmp_path):
    X, y = cancer_data
    rankings = feature_ranking(X, y, task="classification", choices=["mi"])
    save_file = tmp_path / "votes.csv"
    voting(rankings, save=True, save_path=str(save_file))
    assert save_file.exists()
