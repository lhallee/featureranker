"""L1 regression ranking: the entry alpha of each feature on the lasso path.

A feature's score is the largest penalty at which its coefficient is still
nonzero, so features that survive stronger shrinkage rank higher. The exact
route computes the true path breakpoints from a precomputed Gram matrix; the
grid route brackets them on a dense warm-started coordinate-descent path.
"""

import logging
import warnings

import numpy as np

from dataclasses import dataclass
from typing import Literal

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path_gram, lasso_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LassoOptions:
    """Options for the L1 regression ranker.

    The auto strategy picks the exact path when the Gram matrix is affordable:
    p <= max_gram_features caps its memory (p^2 float64) and
    n * p^2 <= max_gram_flops caps the BLAS time to build it.
    """

    strategy: Literal["auto", "exact", "grid"] = "auto"
    n_alphas: int = 200
    eps: float = 1e-4
    tol: float = 1e-6
    max_gram_features: int = 4096
    max_gram_flops: float = 4e12


def _choose_strategy(n_samples: int, n_features: int, options: LassoOptions) -> str:
    if options.strategy != "auto":
        return options.strategy
    affordable = (
        n_features <= options.max_gram_features
        and float(n_samples) * n_features * n_features <= options.max_gram_flops
    )
    return "exact" if affordable else "grid"


def _entry_alphas(
    alphas: np.ndarray, coefs: np.ndarray, bracket: Literal["knot", "geometric"]
) -> np.ndarray:
    """Entry threshold per feature from a decreasing-alpha path.

    "knot" reads the exact breakpoint (the knot before the first nonzero
    coefficient); "geometric" takes the geometric mean of the bracketing grid
    points. Features that never enter score 0.0.
    """
    # alphas: (k,) decreasing; coefs: (p, k)
    nonzero = coefs != 0  # (p, k)
    entered = nonzero.any(axis=1)  # (p,)
    first = nonzero.argmax(axis=1)  # (p,)
    previous = np.maximum(first - 1, 0)  # (p,)
    if bracket == "knot":
        entry = alphas[previous]  # (p,)
    else:
        entry = np.sqrt(alphas[previous] * alphas[first])  # (p,)
    return np.where(entered, entry, 0.0)  # (p,)


def _standardized_moments(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardized Gram matrix and feature-target covariance, without copying X.

    Accumulates float64 moments over row chunks of the (possibly float32)
    shared array, then standardizes by algebra: for centered, unit-variance
    columns, Gs = (G - n mu mu^T) / (sd sd^T) and Xy = (X^T y - n mu ybar) / sd.
    """
    n, p = X.shape
    G = np.zeros((p, p), dtype=np.float64)  # (p, p)
    column_sums = np.zeros(p, dtype=np.float64)  # (p,)
    Xty = np.zeros(p, dtype=np.float64)  # (p,)

    rows_per_chunk = max(1, int(2.56e8 / (p * 8)))
    for start in range(0, n, rows_per_chunk):
        chunk = X[start : start + rows_per_chunk].astype(np.float64)  # (c, p)
        G += chunk.T @ chunk
        column_sums += chunk.sum(axis=0)
        Xty += chunk.T @ y[start : start + rows_per_chunk]

    mu = column_sums / n  # (p,)
    variance = G.diagonal() / n - mu * mu  # (p,)
    sd = np.sqrt(np.clip(variance, 0.0, None))  # (p,)
    sd[sd == 0.0] = 1.0

    Gs = (G - n * np.outer(mu, mu)) / np.outer(sd, sd)  # (p, p)
    Gs = (Gs + Gs.T) / 2.0
    Xy = (Xty - n * mu * y.mean()) / sd  # (p,)
    return Gs, Xy


def _exact_entry_alphas(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Exact path breakpoints via LARS on the standardized Gram matrix."""
    n, p = X.shape
    Gs, Xy = _standardized_moments(X, y)  # (p, p), (p,)
    alphas, _, coefs = lars_path_gram(
        Xy=Xy,
        Gram=Gs,
        n_samples=n,
        method="lasso",
        max_iter=max(1000, 4 * p),
        alpha_min=0.0,
        copy_Gram=False,
    )  # alphas: (k,) decreasing; coefs: (p, k)
    return _entry_alphas(alphas, coefs, bracket="knot"), len(alphas)


def _grid_entry_alphas(
    X: np.ndarray, y: np.ndarray, options: LassoOptions
) -> tuple[np.ndarray, int]:
    """Bracketed entry alphas on a dense warm-started coordinate-descent path."""
    n, p = X.shape
    mu = X.mean(axis=0, dtype=np.float64)  # (p,)
    sd = X.std(axis=0, dtype=np.float64)  # (p,)
    sd[sd == 0.0] = 1.0

    Xs = np.empty((n, p), dtype=X.dtype, order="F")  # (n, p)
    np.subtract(X, mu.astype(X.dtype), out=Xs)
    Xs /= sd.astype(X.dtype)
    y_centered = (y - y.mean()).astype(X.dtype)  # (n,)

    # Entry detection needs the sparsity pattern near each crossing, not the
    # final duality gap; at tol=1e-6 on float32 data the gap check can sit at
    # the precision floor and warn without affecting which entries appear.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        alphas, coefs, _ = lasso_path(
            Xs, y_centered, n_alphas=options.n_alphas, eps=options.eps,
            tol=options.tol,
        )  # alphas: (k,) decreasing; coefs: (p, k)
    return _entry_alphas(alphas, coefs, bracket="geometric"), len(alphas)


def rank_lasso(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budget: int,
    random_state: int,
    options: LassoOptions,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score features by their entry alpha on the L1 regression path."""
    # X: (n, p); y: (n,)
    n, p = X.shape
    strategy = _choose_strategy(n, p, options)

    n_path_points = 0
    scores = None
    if strategy == "exact":
        try:
            scores, n_path_points = _exact_entry_alphas(X, y.astype(np.float64))
        except (np.linalg.LinAlgError, ValueError) as error:
            logger.warning(
                "Exact lasso path failed (%s); falling back to the grid path.", error
            )
            strategy = "grid"
    if scores is None:
        scores, n_path_points = _grid_entry_alphas(X, y, options)

    diagnostics = {
        "strategy": strategy,
        "n_path_points": n_path_points,
        "n_never_entered": int((scores == 0.0).sum()),
    }
    return scores.astype(np.float64), diagnostics
