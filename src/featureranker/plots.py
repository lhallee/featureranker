import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score


def _setup_axes(
    ax: plt.Axes | None, figsize: tuple[float, float]
) -> tuple[plt.Figure | None, plt.Axes]:
    if ax is not None:
        return None, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_correlations(
    predictions: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of predictions vs true values with correlation stats."""
    _, ax = _setup_axes(ax, (10, 6))
    ax.scatter(labels, predictions, alpha=0.5)
    m, b = np.polyfit(labels, predictions, 1)
    ax.plot(labels, m * labels + b, color="red")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title(f"Predictions vs. true values for {model_name}")

    pearson_corr, pearson_pval = stats.pearsonr(labels, predictions)
    spearman_corr, spearman_pval = stats.spearmanr(labels, predictions)
    r2 = r2_score(labels, predictions)
    ax.annotate(
        f"Pearson: {pearson_corr:.2f} (p={pearson_pval:.2e})",
        xy=(0.05, 0.95), xycoords="axes fraction", fontsize=10, verticalalignment="top",
    )
    ax.annotate(
        f"Spearman: {spearman_corr:.2f} (p={spearman_pval:.2e})",
        xy=(0.05, 0.90), xycoords="axes fraction", fontsize=10, verticalalignment="top",
    )
    ax.annotate(
        f"R2: {r2:.2f}",
        xy=(0.05, 0.85), xycoords="axes fraction", fontsize=10, verticalalignment="top",
    )

    if save:
        path = save_path or f"{model_name.replace(' ', '_')}.png"
        plt.savefig(path, bbox_inches="tight", transparent=False, dpi=300)
    if show:
        plt.show()
    return ax


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

    if save:
        path = save_path or f"{title.replace(' ', '_')}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_after_vote(
    scoring: "pd.DataFrame",
    title: str = "Feature Scores",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.25,
    highlight_feature: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar chart of aggregated feature scores."""
    features = scoring["Feature"].tolist()
    scores = scoring["Score"].tolist()
    fig_height = max(len(features) * height_per_feature, 2.0)

    _, ax = _setup_axes(ax, (10, fig_height))
    colors = ["#f0c040" if f == highlight_feature else "#4878d0" for f in features]
    ax.barh(features, scores, color=colors, alpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Scores")
    ax.set_ylabel("Features")
    ax.set_title(title)

    if save:
        path = save_path or f"{title.replace(' ', '_')}.png"
        plt.savefig(path, bbox_inches="tight", transparent=False, dpi=300)
    if show:
        plt.show()
    return ax


def plot_rankings(
    rankings: list[tuple[str, "pd.DataFrame"]],
    title: str = "Feature Rankings",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.25,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlapping horizontal bar charts for multiple ranking methods."""
    assert rankings, "Rankings list is empty."
    n_features = len(rankings[0][1])
    fig_height = max(n_features * height_per_feature, 2.0)
    colors = ["#4878d0", "#6acc65", "#d65f5f", "#f0c040", "#72bcd4", "#ab63fa", "#ff7f0e"]

    _, ax = _setup_axes(ax, (10, fig_height))
    for i, (ranking_name, ranking_df) in enumerate(rankings):
        features = ranking_df[ranking_name].tolist()
        scores = list(reversed(range(1, len(features) + 1)))
        ax.barh(
            features, scores,
            color=colors[i % len(colors)],
            alpha=0.3,
            label=ranking_name,
            edgecolor="black",
        )

    ax.invert_yaxis()
    ax.set_xlabel("Scores")
    ax.set_ylabel("Features")
    ax.set_title(title)
    ax.legend(title="Rankings", bbox_to_anchor=(1.05, 1), loc="upper left")

    if save:
        path = save_path or f"{title.replace(' ', '_')}.png"
        plt.savefig(path, bbox_inches="tight", transparent=False, dpi=300)
    if show:
        plt.show()
    return ax
