"""End-to-end integration tests for the full featureranker pipeline."""

import matplotlib
matplotlib.use("Agg")

from featureranker import feature_ranking, get_data, view_data, voting
from featureranker.plots import plot_after_vote, plot_rankings


def test_classification_pipeline(cancer_data):
    X, y = cancer_data
    rankings = feature_ranking(
        X, y,
        task="classification",
        choices=["mi", "f_test", "l1"],
    )
    assert len(rankings) == 3
    scoring = voting(rankings)
    assert len(scoring) == X.shape[1]
    ax = plot_after_vote(scoring, show=False)
    assert ax is not None
    ax = plot_rankings(rankings, show=False)
    assert ax is not None


def test_regression_pipeline(diabetes_data):
    X, y = diabetes_data
    rankings = feature_ranking(
        X, y,
        task="regression",
        choices=["mi", "f_test", "l1"],
    )
    assert len(rankings) == 3
    scoring = voting(rankings, method="borda")
    assert len(scoring) == X.shape[1]


def test_full_pipeline_from_dataframe(messy_df):
    nan_pct = view_data(messy_df)
    assert "mostly_nan" in nan_pct.index

    X, y = get_data(messy_df, target="target")
    assert "constant" not in X.columns
    assert "mostly_nan" not in X.columns

    rankings = feature_ranking(
        X, y,
        task="classification",
        choices=["mi", "f_test"],
    )
    scoring = voting(rankings)
    assert len(scoring) == X.shape[1]


def test_multiclass_pipeline(iris_data):
    X, y = iris_data
    rankings = feature_ranking(
        X, y,
        task="classification",
        choices=["mi", "f_test"],
    )
    scoring = voting(rankings, method="exponential")
    assert len(scoring) == X.shape[1]
