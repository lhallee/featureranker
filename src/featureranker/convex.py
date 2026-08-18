"""Optimal convex combination of features: weights >= 0 that sum to one."""

import logging

import numpy as np
import pandas as pd

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from scipy.optimize import minimize
from sklearn.metrics import r2_score, roc_auc_score

from .ranking import _convert_features, _convert_target
from .vote import voting

if TYPE_CHECKING:
    from .result import RankingResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class ConvexFit:
    """A fitted convex combination: score(x) = sum_i weights_i * x_i.

    Weights are nonnegative and sum to one, so the score is a weighted
    average of the (optionally standardized) features and each weight reads
    as that feature's share of the combination. When the fit standardized,
    feature_means/feature_stds hold the stored transform and predict
    applies it; both are None for a raw fit. metric_value holds R2 for
    regression and ROC AUC for binary classification, both computed on the
    fitting data.
    """

    task: str
    feature_names: tuple[str, ...]
    weights: np.ndarray  # (k,) nonnegative, sums to one
    feature_means: np.ndarray | None  # (k,) or None for a raw fit
    feature_stds: np.ndarray | None  # (k,) or None for a raw fit
    metric_name: str
    metric_value: float
    diagnostics: dict[str, object]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Score rows with the fitted combination.

        A DataFrame is matched by column name; a numpy array must carry
        exactly the fitted features, in feature_names order. A standardized
        fit applies its stored feature transform first.
        """
        if isinstance(X, pd.DataFrame):
            named = X.rename(columns=str)
            missing = [f for f in self.feature_names if f not in named.columns]
            if missing:
                raise ValueError(f"X is missing fitted features: {missing}.")
            X_sel = named[list(self.feature_names)].to_numpy(dtype=np.float64)  # (n, k)
        else:
            X_sel = np.asarray(X, dtype=np.float64)  # (n, k)
            if X_sel.ndim != 2 or X_sel.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"A numpy X must be 2D with {len(self.feature_names)} columns, "
                    f"got shape {X_sel.shape}."
                )
        if self.feature_means is not None:
            X_sel = (X_sel - self.feature_means) / self.feature_stds  # (n, k)
        return X_sel @ self.weights  # (n,)

    def table(self) -> pd.DataFrame:
        """Weights as ["feature", "weight"], largest first, name as tiebreak."""
        table = pd.DataFrame({"feature": self.feature_names, "weight": self.weights})
        table = table.sort_values(
            ["weight", "feature"], ascending=[False, True], kind="mergesort"
        )
        return table.reset_index(drop=True)

    def __repr__(self) -> str:
        return (
            f"ConvexFit(task={self.task!r}, n_features={len(self.feature_names)}, "
            f"{self.metric_name}={self.metric_value:.4f})"
        )


def _column_scales(X: np.ndarray) -> np.ndarray:
    """Per-column standard deviations with constant columns mapped to 1."""
    scales = X.std(axis=0)  # (k,)
    scales[scales == 0.0] = 1.0
    return scales


def _solve_simplex_least_squares(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Minimize mean squared error of X @ w against y over the simplex.

    The problem is a convex quadratic program, so the solution from the
    uniform start is the global optimum and the fit is deterministic.
    Solving in the substituted variables v = w * scales keeps the Hessian
    well conditioned when column scales differ by orders of magnitude,
    which otherwise stalls SLSQP at the start; the constraint set is still
    the raw simplex, so the returned weights solve the original problem.
    """
    # X: (n, k); y: (n,)
    n, k = X.shape
    scales = _column_scales(X)  # (k,)
    A = X / scales  # (n, k)
    inv_scales = 1.0 / scales  # (k,)
    v0 = scales / k  # (k,), the uniform w0 = 1/k in substituted variables

    def objective(v: np.ndarray) -> float:
        residual = A @ v - y  # (n,)
        return 0.5 * float(residual @ residual) / n

    def gradient(v: np.ndarray) -> np.ndarray:
        residual = A @ v - y  # (n,)
        return A.T @ residual / n  # (k,)

    solution = minimize(
        objective,
        v0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, scale) for scale in scales],
        constraints=[{
            "type": "eq",
            "fun": lambda v: float(v @ inv_scales) - 1.0,
            "jac": lambda v: inv_scales,
        }],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not solution.success:
        logger.warning("Simplex solver stopped early: %s", solution.message)

    # map back to w and clean up round-off so the constraint holds exactly;
    # weights below 1e-12 on a simplex are numerically zero
    weights = np.clip(solution.x / scales, 0.0, None)  # (k,)
    weights[weights < 1e-12] = 0.0
    weights = weights / weights.sum()
    diagnostics: dict[str, object] = {
        "converged": bool(solution.success),
        "iterations": int(solution.nit),
        "mse": float(np.mean((X @ weights - y) ** 2)),
    }
    return weights, diagnostics


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    task: str,
    standardize: bool,
) -> ConvexFit:
    """Solve the simplex fit and package it with its fitting-data metric."""
    # X: (n, k); y: (n,)
    if standardize:
        feature_means = X.mean(axis=0)  # (k,)
        feature_stds = _column_scales(X)  # (k,)
        X_fit = (X - feature_means) / feature_stds  # (n, k)
    else:
        feature_means = feature_stds = None
        X_fit = X  # (n, k)

    weights, diagnostics = _solve_simplex_least_squares(X_fit, y)
    scores = X_fit @ weights  # (n,)
    if task == "classification":
        metric_name, metric_value = "auc", float(roc_auc_score(y, scores))
    else:
        metric_name, metric_value = "r2", float(r2_score(y, scores))
    logger.info(
        "Convex fit over %d features: %s=%.4f.",
        len(feature_names), metric_name, metric_value,
    )
    return ConvexFit(
        task=task,
        feature_names=feature_names,
        weights=weights,
        feature_means=feature_means,
        feature_stds=feature_stds,
        metric_name=metric_name,
        metric_value=metric_value,
        diagnostics=diagnostics,
    )


def _binary_target(
    y: pd.Series | np.ndarray, task: str, n_samples: int
) -> np.ndarray:
    """Convert the target for a simplex fit; classification must be binary."""
    y_arr, classes = _convert_target(y, task, n_samples)  # (n,)
    if classes is not None and len(classes) != 2:
        raise ValueError(
            f"Convex fitting supports binary classification; y has "
            f"{len(classes)} classes. A single score cannot separate more."
        )
    return y_arr.astype(np.float64)  # (n,)


def fit_convex(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    standardize: bool = True,
) -> ConvexFit:
    """Fit the optimal convex combination of every column of X against y.

    Finds weights >= 0 summing to one that minimize the squared error of
    the combined score against y (for classification, against the 0/1
    encoded labels, so the score ranks rows by class membership rather
    than calibrated probability). With standardize=True features are
    z-scored first, making the weights unit-free shares; that is the right
    setting when features carry different units or scales. Pass
    standardize=False when the features are already commensurate (for
    example sub-scores of a ranking scheme) and the score should be the
    weighted average of the raw values. To combine only the strongest
    features from a ranking run, call RankingResult.fit_convex with top_n
    instead.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task {task!r}. Valid: 'classification', 'regression'.")
    X_arr, feature_names = _convert_features(X, "float64")  # (n, k)
    y_arr = _binary_target(y, task, X_arr.shape[0])  # (n,)
    return _fit(X_arr, y_arr, feature_names, task, standardize)


def fit_convex_from_result(
    result: "RankingResult",
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    top_n: int | None = None,
    weights: Mapping[str, float] | None = None,
    vote_method: Literal["reciprocal_rank", "borda", "exponential"] = "reciprocal_rank",
    standardize: bool = True,
) -> ConvexFit:
    """Fit a convex combination of the top consensus features of a ranking.

    Consensus order comes from voting(result, weights, vote_method); the
    top_n features (all when None) are fitted. X must carry the same
    features that produced the ranking result, though the rows may differ.
    standardize behaves as in fit_convex.
    """
    X_arr, feature_names = _convert_features(X, "float64")  # (n, p)
    if feature_names != result.feature_names:
        raise ValueError(
            "X does not match the ranked features; pass a matrix with the "
            "same columns that produced this RankingResult."
        )
    y_arr = _binary_target(y, result.task, X_arr.shape[0])  # (n,)

    if top_n is None:
        top_n = result.n_features
    if not 1 <= top_n <= result.n_features:
        raise ValueError(f"top_n must be in [1, {result.n_features}], got {top_n}.")

    consensus = voting(result, weights=weights, method=vote_method)
    selected = tuple(consensus["feature"].head(top_n))  # k names, best first
    column_index = {name: i for i, name in enumerate(feature_names)}
    X_sel = X_arr[:, [column_index[name] for name in selected]]  # (n, k)
    return _fit(X_sel, y_arr, selected, result.task, standardize)
