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
from .utils import *
from .rankers import *
from .plots import *
from .clustering import *
