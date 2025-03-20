"""
Feature Ranker - A package for ranking features using multiple methods.

This package provides functions for:
- Data preprocessing and visualization
- Feature ranking using various methods
- Voting aggregation of feature rankings
- Visualization of feature importances and model performance
- Feature clustering capabilities
"""

__version__ = "1.2.2"

# Import key functions for easier access
from .utils import get_data, view_data, sanitize_column_names
from .rankers import classification_ranking, regression_ranking, voting
from .plots import plot_after_vote, plot_rankings, plot_correlations, plot_confusion_matrix
from .clustering import optimal_k_w_elbow, optimal_k_w_both, random_cluster_generator
