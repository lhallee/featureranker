"""Mutual information and ANOVA F-test rankers."""

import logging
import warnings

import numpy as np

from dataclasses import dataclass
from scipy.stats import ConstantInputWarning
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MutualInfoOptions:
    """Options for the mutual information ranker.

    max_samples caps the rows used by the kNN estimator; above it a seeded
    subsample is drawn (the estimate stabilizes well below the cap while the
    exact cost grows as n log n per feature). None disables subsampling.
    """

    n_neighbors: int = 3
    max_samples: int | None = 100_000


def rank_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budget: int,
    random_state: int,
    options: MutualInfoOptions,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score features by estimated mutual information with the target."""
    # X: (n, p); y: (n,)
    n = X.shape[0]
    subsampled = options.max_samples is not None and n > options.max_samples
    if subsampled:
        rng = np.random.default_rng(random_state)
        rows = np.sort(rng.choice(n, size=options.max_samples, replace=False))  # (max_samples,)
        X = X[rows]  # (max_samples, p)
        y = y[rows]  # (max_samples,)
        logger.info("Mutual information subsampled %d of %d rows.", len(rows), n)

    mi = mutual_info_classif if task == "classification" else mutual_info_regression
    scores = mi(
        X,
        y,
        n_neighbors=options.n_neighbors,
        random_state=random_state,
        n_jobs=budget,
    )  # (p,)
    diagnostics = {
        "n_neighbors": options.n_neighbors,
        "n_samples_used": len(y),
        "subsampled": subsampled,
    }
    return scores.astype(np.float64), diagnostics


def rank_f_test(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budget: int,
    random_state: int,
    options: None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score features by the ANOVA F statistic against the target."""
    # X: (n, p); y: (n,)
    if task == "classification":
        # Constant features produce NaN scores and a warning; they rank last.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            scores, _ = f_classif(X, y)  # (p,)
        scores = np.nan_to_num(scores, nan=0.0)
    else:
        scores, _ = f_regression(X, y, force_finite=True)  # (p,)
    return scores.astype(np.float64), {}
