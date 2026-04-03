import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from featureranker.plots import (
    plot_after_vote,
    plot_confusion_matrix,
    plot_correlations,
    plot_rankings,
)


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


def test_plot_correlations():
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    ax = plot_correlations(preds, labels, "TestModel", show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_confusion_matrix():
    cm = np.array([[10, 2], [3, 15]])
    labels = np.array([0, 1])
    ax = plot_confusion_matrix(cm, labels, title="Test CM", show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_after_vote():
    scoring = pd.DataFrame({"Feature": ["a", "b", "c"], "Score": [0.5, 0.3, 0.1]})
    ax = plot_after_vote(scoring, title="Test Vote", show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_after_vote_highlight():
    scoring = pd.DataFrame({"Feature": ["a", "b", "c"], "Score": [0.5, 0.3, 0.1]})
    ax = plot_after_vote(scoring, highlight_feature="b", show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_rankings():
    r1 = pd.DataFrame({"RF": ["a", "b", "c"], "Score": [3, 2, 1]})
    r2 = pd.DataFrame({"MI": ["b", "a", "c"], "Score": [3, 2, 1]})
    rankings = [("RF", r1), ("MI", r2)]
    ax = plot_rankings(rankings, title="Test Rankings", show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_correlations_save(tmp_path):
    labels = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.1, 2.0, 3.1])
    path = tmp_path / "corr.png"
    plot_correlations(preds, labels, "M", save=True, save_path=str(path), show=False)
    assert path.exists()


def test_plot_confusion_matrix_save(tmp_path):
    cm = np.array([[5, 1], [2, 8]])
    path = tmp_path / "cm.png"
    plot_confusion_matrix(cm, np.array([0, 1]), save=True, save_path=str(path), show=False)
    assert path.exists()
