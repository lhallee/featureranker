"""Tests for the plot functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from featureranker.plots import (
    plot_after_vote,
    plot_confusion_matrix,
    plot_correlations,
    plot_rankings,
)
from featureranker.ranking import feature_ranking
from featureranker.vote import voting


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def univariate_result(synthetic_classification):
    X, y = synthetic_classification
    return feature_ranking(X, y, methods=["mi", "f_test"])


def test_plot_correlations():
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    ax = plot_correlations(predictions, labels, "TestModel", show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_confusion_matrix():
    c_matrix = np.array([[10, 2], [3, 15]])
    ax = plot_confusion_matrix(c_matrix, np.array([0, 1]), show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_after_vote(univariate_result):
    ax = plot_after_vote(voting(univariate_result), show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_after_vote_highlight():
    vote_table = pd.DataFrame({"feature": ["a", "b", "c"], "score": [0.5, 0.3, 0.1]})
    ax = plot_after_vote(vote_table, highlight_feature="b", show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_rankings_from_result(univariate_result):
    ax = plot_rankings(univariate_result, show=False)
    assert isinstance(ax, plt.Axes)
    assert len(ax.get_yticklabels()) == univariate_result.n_features


def test_plot_rankings_top_n(univariate_result):
    ax = plot_rankings(univariate_result, top_n=5, show=False)
    assert len(ax.get_yticklabels()) == 5


def test_plot_rankings_save(univariate_result, tmp_path):
    path = tmp_path / "rankings.png"
    plot_rankings(univariate_result, save=True, save_path=str(path), show=False)
    assert path.exists()


def test_plot_correlations_save(tmp_path):
    labels = np.array([1.0, 2.0, 3.0])
    predictions = np.array([1.1, 2.0, 3.1])
    path = tmp_path / "corr.png"
    plot_correlations(predictions, labels, "M", save=True, save_path=str(path), show=False)
    assert path.exists()


def test_save_uses_axes_figure_not_current(tmp_path):
    """v2 saved plt.gcf(); passing an ax from a non-current figure must work."""
    vote_table = pd.DataFrame({"feature": ["a", "b"], "score": [0.9, 0.1]})
    small_fig, small_ax = plt.subplots(figsize=(3, 2))
    big_fig, _ = plt.subplots(figsize=(12, 9))
    assert plt.gcf() is big_fig

    path = tmp_path / "which.png"
    plot_after_vote(vote_table, save=True, save_path=str(path), show=False, ax=small_ax)
    saved = plt.imread(path)  # (h, w, 4)
    # the 3x2-inch figure saves near 900x600 px; the 12x9 current figure would
    # be near 3600x2700
    assert saved.shape[1] < 1500, "saved the wrong (current) figure"
