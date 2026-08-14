"""End-to-end tests for the full featureranker pipeline."""

from featureranker import feature_ranking, get_data, view_data, voting
from featureranker.plots import plot_after_vote, plot_rankings


def test_classification_pipeline(cancer_data):
    X, y = cancer_data
    result = feature_ranking(X, y, methods=["mi", "f_test", "l1"])
    assert result.methods == ("mi", "f_test", "l1")
    vote_table = voting(result)
    assert len(vote_table) == X.shape[1]
    assert plot_after_vote(vote_table, show=False) is not None
    assert plot_rankings(result, show=False) is not None


def test_regression_pipeline(diabetes_data):
    X, y = diabetes_data
    result = feature_ranking(X, y, task="regression", methods=["mi", "f_test", "l1"])
    vote_table = voting(result, method="borda")
    assert len(vote_table) == X.shape[1]


def test_full_pipeline_from_dataframe(messy_df):
    nan_pct = view_data(messy_df)
    assert "mostly_nan" in nan_pct.index

    X, y = get_data(messy_df, target="target")
    assert "constant" not in X.columns
    assert "mostly_nan" not in X.columns

    result = feature_ranking(X, y, methods=["mi", "f_test"])
    vote_table = voting(result)
    assert len(vote_table) == X.shape[1]


def test_multiclass_pipeline(iris_data):
    """Includes l1: the v2 L1 ranker crashed on 3+ classes."""
    X, y = iris_data
    result = feature_ranking(X, y, methods=["mi", "f_test", "l1"])
    assert result.diagnostics["l1"]["solver"] == "saga"
    vote_table = voting(result, method="exponential")
    assert len(vote_table) == X.shape[1]


def test_full_ensemble_with_weights(synthetic_classification):
    X, y = synthetic_classification
    result = feature_ranking(X, y)
    vote_table = voting(result, weights={"rf": 2.0, "l1": 0.5})
    top4 = set(vote_table["feature"].head(4))
    assert top4 == {"feat_0", "feat_1", "feat_2", "feat_3"}
