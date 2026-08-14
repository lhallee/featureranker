"""Plots for ranking results and model evaluation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.metrics import r2_score

from .result import RankingResult

_BAR_COLORS = ["#4878d0", "#6acc65", "#d65f5f", "#f0c040", "#72bcd4", "#ab63fa", "#ff7f0e"]
_HIGHLIGHT_COLOR = "#f0c040"
_BASE_COLOR = "#4878d0"


def _setup_axes(
    ax: plt.Axes | None, figsize: tuple[float, float]
) -> tuple[plt.Figure | None, plt.Axes]:
    if ax is not None:
        return None, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finish(
    ax: plt.Axes, save: bool, save_path: str | None, default_name: str, show: bool
) -> plt.Axes:
    if save:
        path = save_path or f"{default_name.replace(' ', '_')}.png"
        # save the axes' own figure: plt.savefig would grab the current one
        ax.figure.savefig(path, dpi=300, bbox_inches="tight", transparent=False)
    if show:
        plt.show()
    return ax


def plot_rankings(
    result: RankingResult,
    top_n: int | None = 30,
    title: str = "Feature Rankings",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.25,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlaid per-method inverse ranks for the top features.

    Features are ordered by their average rank across methods; each bar shows
    n_features - rank + 1, so longer means better in that method.
    """
    R = result.rank_matrix()  # (p, m)
    order = R.mean(axis=1).sort_values(kind="stable").index
    if top_n is not None:
        order = order[:top_n]

    _, ax = _setup_axes(ax, (10, max(len(order) * height_per_feature, 2.0)))
    for i, method in enumerate(R.columns):
        inverse_rank = result.n_features - R.loc[order, method] + 1  # (top_n,)
        ax.barh(
            list(order),
            inverse_rank,
            color=_BAR_COLORS[i % len(_BAR_COLORS)],
            alpha=0.35,
            label=method,
            edgecolor="black",
        )
    ax.invert_yaxis()
    ax.set_xlabel("Inverse rank (higher is better)")
    ax.set_ylabel("Features")
    ax.set_title(title)
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    return _finish(ax, save, save_path, title, show)


def plot_after_vote(
    vote_table: pd.DataFrame,
    top_n: int | None = 30,
    title: str = "Feature Scores",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.25,
    highlight_feature: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bars of aggregated vote scores from voting()."""
    table = vote_table.head(top_n) if top_n is not None else vote_table
    features = table["feature"].tolist()
    scores = table["score"].tolist()

    _, ax = _setup_axes(ax, (10, max(len(features) * height_per_feature, 2.0)))
    colors = [
        _HIGHLIGHT_COLOR if feature == highlight_feature else _BASE_COLOR
        for feature in features
    ]
    ax.barh(features, scores, color=colors, alpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Vote score")
    ax.set_ylabel("Features")
    ax.set_title(title)
    return _finish(ax, save, save_path, title, show)


def plot_correlations(
    predictions: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of predictions against true values with correlation statistics."""
    # predictions, labels: (n,)
    _, ax = _setup_axes(ax, (10, 6))
    ax.scatter(labels, predictions, alpha=0.5)
    slope, intercept = np.polyfit(labels, predictions, 1)
    ax.plot(labels, slope * labels + intercept, color="red")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predictions")
    ax.set_title(f"Predictions vs. true values for {model_name}")

    pearson, pearson_p = stats.pearsonr(labels, predictions)
    spearman, spearman_p = stats.spearmanr(labels, predictions)
    r2 = r2_score(labels, predictions)
    for offset, text in enumerate(
        (
            f"Pearson: {pearson:.2f} (p={pearson_p:.2e})",
            f"Spearman: {spearman:.2f} (p={spearman_p:.2e})",
            f"R2: {r2:.2f}",
        )
    ):
        ax.annotate(
            text,
            xy=(0.05, 0.95 - 0.05 * offset),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
        )
    return _finish(ax, save, save_path, model_name, show)


def plot_confusion_matrix(
    c_matrix: np.ndarray,
    labels: np.ndarray,
    title: str = "Confusion Matrix",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of a confusion matrix."""
    # c_matrix: (k, k); labels: (k,)
    _, ax = _setup_axes(ax, (8, 6))
    sns.heatmap(
        c_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Actual label")
    ax.set_xlabel("Predicted label")
    return _finish(ax, save, save_path, title, show)
