"""featureranker - Feature ranking ensemble using multiple methods with weighted voting."""

__version__ = "2.0.0"

from .clustering import (
    get_inertia,
    get_kmean_metrics,
    optimal_k_w_both,
    optimal_k_w_elbow,
    random_cluster_generator,
)
from .plots import (
    plot_after_vote,
    plot_confusion_matrix,
    plot_correlations,
    plot_rankings,
)
from .rankers import (
    VALID_CHOICES,
    feature_ranking,
    l1_classification_ranking,
    l1_regression_ranking,
    make_ranking,
    voting,
)
from .utils import (
    get_data,
    hyper_param_search,
    sanitize_column_names,
    view_data,
)

__all__ = [
    # rankers
    "feature_ranking",
    "voting",
    "make_ranking",
    "l1_regression_ranking",
    "l1_classification_ranking",
    "VALID_CHOICES",
    # utils
    "get_data",
    "view_data",
    "sanitize_column_names",
    "hyper_param_search",
    # plots
    "plot_correlations",
    "plot_confusion_matrix",
    "plot_after_vote",
    "plot_rankings",
    # clustering
    "random_cluster_generator",
    "get_inertia",
    "optimal_k_w_elbow",
    "get_kmean_metrics",
    "optimal_k_w_both",
]
