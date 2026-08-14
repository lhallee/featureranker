"""featureranker: ensemble feature ranking with weighted voting."""

__version__ = "2.0.0"

from .data import get_data, sanitize_column_names, view_data
from .lasso import LassoOptions
from .logistic import LogisticL1Options
from .plots import (
    plot_after_vote,
    plot_confusion_matrix,
    plot_correlations,
    plot_rank_heatmap,
    plot_rankings,
)
from .ranking import METHODS, feature_ranking
from .result import RankingResult
from .trees import TreeSearchOptions
from .univariate import MutualInfoOptions
from .vote import voting

__all__ = [
    # ranking
    "feature_ranking",
    "RankingResult",
    "voting",
    "METHODS",
    # options
    "LassoOptions",
    "LogisticL1Options",
    "TreeSearchOptions",
    "MutualInfoOptions",
    # data preparation
    "get_data",
    "view_data",
    "sanitize_column_names",
    # plots
    "plot_rankings",
    "plot_rank_heatmap",
    "plot_after_vote",
    "plot_correlations",
    "plot_confusion_matrix",
]
