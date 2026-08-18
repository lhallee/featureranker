"""featureranker: ensemble feature ranking with weighted voting."""

__version__ = "3.0.1"

from .convex import ConvexFit, fit_convex
from .data import get_data, sanitize_column_names, view_data
from .hf import get_hf_data, hf_login, load_hf_dataset
from .lasso import LassoOptions
from .logistic import LogisticL1Options
from .plots import (
    plot_after_vote,
    plot_confusion_matrix,
    plot_correlations,
    plot_rank_heatmap,
    plot_rankings,
)
from .ranking import METHODS, feature_ranking, generated_feature_names
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
    "generated_feature_names",
    # convex combination
    "fit_convex",
    "ConvexFit",
    # options
    "LassoOptions",
    "LogisticL1Options",
    "TreeSearchOptions",
    "MutualInfoOptions",
    # data preparation
    "get_data",
    "view_data",
    "sanitize_column_names",
    # hugging face
    "get_hf_data",
    "load_hf_dataset",
    "hf_login",
    # plots
    "plot_rankings",
    "plot_rank_heatmap",
    "plot_after_vote",
    "plot_correlations",
    "plot_confusion_matrix",
]
