"""Plots for ranking results and model evaluation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.metrics import r2_score

from .result import RankingResult

# fixed identity colors: a method keeps its hue regardless of which methods run
_METHOD_COLORS = {
    "rf": "#2a78d6",
    "xg": "#eb6834",
    "mi": "#1baf7a",
    "f_test": "#eda100",
    "l1": "#e87ba4",
}
_EXTRA_COLORS = ["#008300", "#4a3aa7", "#e34948"]
_METHOD_LABELS = {
    "rf": "Random forest",
    "xg": "XGBoost",
    "mi": "Mutual information",
    "f_test": "F-test",
    "l1": "L1 path",
}

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_ACCENT = "#2a78d6"
_DEEMPHASIS = "#c3c2b7"
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

_SEQUENTIAL = LinearSegmentedColormap.from_list("featureranker_blues", _BLUE_RAMP)
_SEQUENTIAL_R = LinearSegmentedColormap.from_list(
    "featureranker_blues_r", list(reversed(_BLUE_RAMP))
)


def _method_color(method: str, fallback_index: int) -> str:
    if method in _METHOD_COLORS:
        return _METHOD_COLORS[method]
    return _EXTRA_COLORS[fallback_index % len(_EXTRA_COLORS)]


def _setup_axes(
    ax: plt.Axes | None, figsize: tuple[float, float]
) -> tuple[plt.Figure | None, plt.Axes]:
    if ax is not None:
        return None, ax
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(_SURFACE)
    return fig, ax


def _style_axes(ax: plt.Axes, title: str, baseline: str = "left") -> None:
    """Recessive chrome: one baseline spine, hairline grid, ink-token text."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side == baseline)
    ax.spines[baseline].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_SECONDARY, length=0)
    ax.set_title(title, loc="left", color=_INK, fontsize=12, fontweight="semibold", pad=14)
    ax.set_axisbelow(True)


def _finish(
    ax: plt.Axes, save: bool, save_path: str | None, default_name: str, show: bool
) -> plt.Axes:
    if save:
        path = save_path or f"{default_name.replace(' ', '_')}.png"
        # save the axes' own figure: plt.savefig would grab the current one
        ax.figure.savefig(
            path, dpi=300, bbox_inches="tight", facecolor=ax.figure.get_facecolor()
        )
    if show:
        plt.show()
    return ax


def _consensus_order(result: RankingResult, top_n: int | None) -> pd.Index:
    """Feature order by average rank across methods, best first."""
    R = result.rank_matrix()  # (p, m)
    order = R.mean(axis=1).sort_values(kind="stable").index
    return order if top_n is None else order[:top_n]


def plot_rankings(
    result: RankingResult,
    top_n: int | None = 30,
    title: str = "Feature ranks by method",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.32,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dot plot of per-method ranks, one row per feature, best rank left.

    Rows are ordered by average rank, so the top row is the consensus best
    feature; the horizontal spread of a row shows how much the methods
    disagree about that feature.
    """
    R = result.rank_matrix()  # (p, m)
    order = _consensus_order(result, top_n)
    rows = np.arange(len(order))  # (top_n,)

    _, ax = _setup_axes(ax, (9, max(len(order) * height_per_feature, 2.4)))
    _style_axes(ax, title, baseline="bottom")

    spread = R.loc[order]  # (top_n, m)
    ax.hlines(
        rows, spread.min(axis=1), spread.max(axis=1),
        color=_GRID, linewidth=1.5, zorder=2,
    )
    for i, method in enumerate(R.columns):
        ax.scatter(
            spread[method], rows,
            s=80, color=_method_color(method, i),
            label=_METHOD_LABELS.get(method, method),
            edgecolors=_SURFACE, linewidths=1.4, zorder=3,
        )

    ax.set_yticks(rows, labels=list(order))
    ax.invert_yaxis()
    ax.margins(y=0.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0.3, result.n_features + 0.7)
    ax.grid(axis="x", color=_GRID, linewidth=1.0)
    ax.set_xlabel("Rank (1 = most important)", color=_INK_SECONDARY)
    legend = ax.legend(
        title=None, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        labelcolor=_INK_SECONDARY, handletextpad=0.1,
    )
    for text in legend.get_texts():
        text.set_fontsize(10)
    return _finish(ax, save, save_path, title, show)


def plot_rank_heatmap(
    result: RankingResult,
    top_n: int | None = 40,
    title: str = "Rank heatmap",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.3,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Feature-by-method heatmap of ranks; darker means more important.

    Scales to wide feature sets better than the dot plot: cell annotations
    switch off automatically past 200 cells.
    """
    order = _consensus_order(result, top_n)
    ranks = result.rank_matrix().loc[order]  # (top_n, m)
    ranks.columns = [_METHOD_LABELS.get(method, method) for method in ranks.columns]

    _, ax = _setup_axes(ax, (9, max(len(order) * height_per_feature, 2.4)))
    annotate = ranks.size <= 200
    sns.heatmap(
        ranks,
        cmap=_SEQUENTIAL_R,
        vmin=1.0,
        vmax=float(result.n_features),
        annot=annotate,
        fmt=".0f",
        annot_kws={"fontsize": 9},
        linewidths=2.0,
        linecolor=_SURFACE,
        cbar_kws={"label": "Rank (1 = most important)"},
        ax=ax,
    )
    ax.set_title(title, loc="left", color=_INK, fontsize=12, fontweight="semibold", pad=14)
    ax.set_ylabel("")
    ax.tick_params(colors=_MUTED, labelcolor=_INK_SECONDARY, length=0)
    plt.setp(ax.get_xticklabels(), rotation=0)
    colorbar = ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.ax.invert_yaxis()
    return _finish(ax, save, save_path, title, show)


def plot_after_vote(
    vote_table: pd.DataFrame,
    top_n: int | None = 30,
    title: str = "Aggregated feature importance",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.3,
    highlight_feature: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bars of aggregated vote scores from voting().

    With highlight_feature set, that bar keeps the accent color and the rest
    recede to gray, which is the fastest way to point at one feature.
    """
    table = vote_table.head(top_n) if top_n is not None else vote_table
    features = table["feature"].tolist()
    scores = table["score"].tolist()

    _, ax = _setup_axes(ax, (9, max(len(features) * height_per_feature, 2.4)))
    _style_axes(ax, title, baseline="left")

    if highlight_feature is None:
        colors = [_ACCENT] * len(features)
    else:
        colors = [
            _ACCENT if feature == highlight_feature else _DEEMPHASIS
            for feature in features
        ]
    ax.barh(features, scores, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.margins(y=0.02)
    ax.grid(axis="x", color=_GRID, linewidth=1.0)
    ax.set_xlabel("Vote score", color=_INK_SECONDARY)
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
    _, ax = _setup_axes(ax, (8, 6))
    _style_axes(ax, f"Predictions vs. true values, {model_name}", baseline="bottom")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(_BASELINE)

    ax.scatter(labels, predictions, color=_ACCENT, alpha=0.55, s=36, edgecolors="none")
    slope, intercept = np.polyfit(labels, predictions, 1)
    span = np.array([labels.min(), labels.max()])  # (2,)
    ax.plot(span, slope * span + intercept, color=_INK_SECONDARY, linewidth=2.0)
    ax.grid(color=_GRID, linewidth=1.0)
    ax.set_xlabel("True values", color=_INK_SECONDARY)
    ax.set_ylabel("Predictions", color=_INK_SECONDARY)

    pearson, pearson_p = stats.pearsonr(labels, predictions)
    spearman, spearman_p = stats.spearmanr(labels, predictions)
    r2 = r2_score(labels, predictions)
    summary = (
        f"Pearson {pearson:.2f} (p={pearson_p:.1e})\n"
        f"Spearman {spearman:.2f} (p={spearman_p:.1e})\n"
        f"R² {r2:.2f}"
    )
    ax.annotate(
        summary,
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        fontsize=10,
        color=_INK_SECONDARY,
        verticalalignment="top",
    )
    return _finish(ax, save, save_path, model_name, show)


def plot_confusion_matrix(
    c_matrix: np.ndarray,
    labels: np.ndarray,
    title: str = "Confusion matrix",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of a confusion matrix; darker means more samples."""
    # c_matrix: (k, k); labels: (k,)
    _, ax = _setup_axes(ax, (7, 5.5))
    sns.heatmap(
        c_matrix,
        annot=True,
        fmt="d",
        cmap=_SEQUENTIAL,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=2.0,
        linecolor=_SURFACE,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title, loc="left", color=_INK, fontsize=12, fontweight="semibold", pad=14)
    ax.set_ylabel("Actual label", color=_INK_SECONDARY)
    ax.set_xlabel("Predicted label", color=_INK_SECONDARY)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_SECONDARY, length=0)
    return _finish(ax, save, save_path, title, show)
