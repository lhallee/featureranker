"""Optimal convex combination of features: weights >= 0 that sum to one."""

import logging
import warnings

import numpy as np
import pandas as pd

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
    applies it; both are None for a raw fit.

    metrics holds R2 (regression) or ROC AUC (binary classification) per
    split: "train" is always present and computed on the fitting data;
    "valid" and "test" appear when those pairs were passed to the fit. A
    fit from a RankingResult also fills method_metrics: the same per-split
    metrics refit on each ranking method's own top_n selection, with
    "ensemble" the returned voting-consensus fit. Use valid to choose
    between selections and settings; quote test for the final result.
    """

    task: str
    feature_names: tuple[str, ...]
    weights: np.ndarray  # (k,) nonnegative, sums to one
    feature_means: np.ndarray | None  # (k,) or None for a raw fit
    feature_stds: np.ndarray | None  # (k,) or None for a raw fit
    metric_name: str
    metrics: dict[str, float]
    diagnostics: dict[str, object]
    method_metrics: dict[str, dict[str, float]] | None = None

    @property
    def metric_value(self) -> float:
        """The fitting-data metric; metrics carries valid/test beside it."""
        return self.metrics["train"]

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
        reported = " ".join(
            f"{split}={value:.4f}" for split, value in self.metrics.items()
        )
        return (
            f"ConvexFit(task={self.task!r}, n_features={len(self.feature_names)}, "
            f"{self.metric_name}: {reported})"
        )


def _column_scales(X: np.ndarray) -> np.ndarray:
    """Per-column standard deviations with constant columns mapped to 1."""
    scales = X.std(axis=0)  # (k,)
    scales[scales == 0.0] = 1.0
    return scales


def _solve_simplex_least_squares(
    X: np.ndarray, y: np.ndarray, gamma: float
) -> tuple[np.ndarray, dict[str, object]]:
    """Minimize mean squared error of X @ w against y over the simplex.

    With gamma=0 the problem is a convex quadratic program whose optimum
    may sit on the simplex boundary, so redundant features get exact zero
    weights. With gamma > 0 a maximum-entropy term gamma * sum(w * log w)
    is added: the objective becomes strictly convex and the entropy
    gradient diverges at the boundary, so the optimum is unique and every
    weight is strictly positive. Either way the fit from the uniform start
    is deterministic. Solving in the substituted variables v = w * scales
    keeps the Hessian well conditioned when column scales differ by orders
    of magnitude, which otherwise stalls SLSQP at the start; the
    constraint set is still the raw simplex, so the returned weights solve
    the original problem.
    """
    # X: (n, k); y: (n,)
    n, k = X.shape
    scales = _column_scales(X)  # (k,)
    A = X / scales  # (n, k)
    inv_scales = 1.0 / scales  # (k,)
    v0 = scales / k  # (k,), the uniform w0 = 1/k in substituted variables
    floor = 1e-12 if gamma > 0.0 else 0.0

    def objective(v: np.ndarray) -> float:
        residual = A @ v - y  # (n,)
        value = 0.5 * float(residual @ residual) / n
        if gamma > 0.0:
            w = v * inv_scales  # (k,)
            value += gamma * float(np.sum(w * np.log(w)))
        return value

    def gradient(v: np.ndarray) -> np.ndarray:
        residual = A @ v - y  # (n,)
        grad = A.T @ residual / n  # (k,)
        if gamma > 0.0:
            w = v * inv_scales  # (k,)
            grad = grad + gamma * inv_scales * (np.log(w) + 1.0)  # (k,)
        return grad

    with warnings.catch_warnings():
        # SLSQP line searches step slightly outside the bounds and scipy
        # clips them with a RuntimeWarning; expected here, not actionable
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        solution = minimize(
            objective,
            v0,
            jac=gradient,
            method="SLSQP",
            bounds=[(floor * scale, scale) for scale in scales],
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
    # without entropy, weights below 1e-12 on a simplex are numerically zero
    weights = np.clip(solution.x / scales, floor, None)  # (k,)
    if gamma == 0.0:
        weights[weights < 1e-12] = 0.0
    weights = weights / weights.sum()
    diagnostics: dict[str, object] = {
        "converged": bool(solution.success),
        "iterations": int(solution.nit),
        "mse": float(np.mean((X @ weights - y) ** 2)),
        "entropy_weight": gamma,
    }
    return weights, diagnostics


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    task: str,
    standardize: bool,
    entropy: float,
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

    # scaling by the target variance makes the smoothing strength invariant
    # to the units of y
    gamma = entropy * float(y.var())
    weights, diagnostics = _solve_simplex_least_squares(X_fit, y, gamma)
    scores = X_fit @ weights  # (n,)
    metric_name = "auc" if task == "classification" else "r2"
    return ConvexFit(
        task=task,
        feature_names=feature_names,
        weights=weights,
        feature_means=feature_means,
        feature_stds=feature_stds,
        metric_name=metric_name,
        metrics={"train": _metric(task, y, scores)},
        diagnostics=diagnostics,
    )


def _metric(task: str, y: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC for classification, R2 for regression."""
    # y: (n,); scores: (n,)
    if task == "classification":
        return float(roc_auc_score(y, scores))
    return float(r2_score(y, scores))


def _binary_target(
    y: pd.Series | np.ndarray, task: str, n_samples: int
) -> tuple[np.ndarray, tuple[object, ...] | None]:
    """Convert the fit target; classification must be binary."""
    y_arr, classes = _convert_target(y, task, n_samples)  # (n,)
    if classes is not None and len(classes) != 2:
        raise ValueError(
            f"Convex fitting supports binary classification; y has "
            f"{len(classes)} classes. A single score cannot separate more."
        )
    return y_arr.astype(np.float64), classes  # (n,)


def _encode_eval_target(
    y: pd.Series | np.ndarray,
    task: str,
    classes: tuple[object, ...] | None,
    n_samples: int,
    split_name: str,
) -> np.ndarray:
    """Encode an evaluation target with the class mapping of the fit target."""
    y_arr = np.asarray(y).ravel()  # (n,)
    if y_arr.shape[0] != n_samples:
        raise ValueError(
            f"{split_name} X has {n_samples} rows but its y has {y_arr.shape[0]}."
        )
    if pd.isna(y_arr).any():
        raise ValueError(f"{split_name} y contains missing values.")
    if task == "classification":
        encoded_of = {label: float(code) for code, label in enumerate(classes)}
        try:
            return np.array([encoded_of[label] for label in y_arr])  # (n,)
        except KeyError as error:
            raise ValueError(
                f"{split_name} y contains label {error.args[0]!r} that the "
                "fitting target never showed."
            ) from None
    y_float = y_arr.astype(np.float64)  # (n,)
    if not np.isfinite(y_float).all():
        raise ValueError(f"{split_name} y contains infinite values.")
    return y_float


def _check_eval_pair(pair: object, split_name: str) -> tuple[object, object]:
    """An evaluation split must arrive as an (X, y) pair."""
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise TypeError(f"{split_name} must be an (X, y) pair, got {type(pair).__name__}.")
    return pair[0], pair[1]


def _attach_eval_metrics(
    fit: ConvexFit,
    classes: tuple[object, ...] | None,
    evals: dict[str, object],
) -> ConvexFit:
    """Score the fit on evaluation pairs and return it with those metrics."""
    metrics = dict(fit.metrics)
    for split_name, pair in evals.items():
        if pair is None:
            continue
        X_eval, y_eval = _check_eval_pair(pair, split_name)
        scores = fit.predict(X_eval)  # (n_eval,)
        y_arr = _encode_eval_target(
            y_eval, fit.task, classes, scores.shape[0], split_name
        )  # (n_eval,)
        metrics[split_name] = _metric(fit.task, y_arr, scores)
    return replace(fit, metrics=metrics)


def fit_convex(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    standardize: bool = True,
    entropy: float = 0.1,
    valid: tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray] | None = None,
    test: tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray] | None = None,
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
    weighted average of the raw values.

    entropy sets the maximum-entropy smoothing strength: the default keeps
    every weight strictly positive, because the entropy gradient diverges
    at the simplex boundary, and makes the optimum unique even when
    features duplicate each other. entropy=0 recovers the plain least
    squares fit, where redundant features get exact zero weights.

    valid and test are optional held-out (X, y) pairs carrying the same
    features; the fit never trains on them and reports their metrics in
    fit.metrics beside "train". To combine only the strongest features
    from a ranking run, call RankingResult.fit_convex with top_n instead.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task {task!r}. Valid: 'classification', 'regression'.")
    if entropy < 0.0:
        raise ValueError(f"entropy must be >= 0, got {entropy}.")
    X_arr, feature_names = _convert_features(X, "float64")  # (n, k)
    y_arr, classes = _binary_target(y, task, X_arr.shape[0])  # (n,)
    fit = _fit(X_arr, y_arr, feature_names, task, standardize, entropy)
    fit = _attach_eval_metrics(fit, classes, {"valid": valid, "test": test})
    logger.info(
        "Convex fit over %d features, %s: %s.",
        len(feature_names), fit.metric_name,
        ", ".join(f"{split}={value:.4f}" for split, value in fit.metrics.items()),
    )
    return fit


def fit_convex_from_result(
    result: "RankingResult",
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    top_n: int | None = None,
    weights: Mapping[str, float] | Literal["auto"] | None = None,
    vote_method: Literal["reciprocal_rank", "borda", "exponential"] = "reciprocal_rank",
    standardize: bool = True,
    entropy: float = 0.1,
    valid: tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray] | None = None,
    test: tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray] | None = None,
) -> ConvexFit:
    """Fit a convex combination of the top consensus features of a ranking.

    Consensus order comes from voting(result, weights, vote_method); the
    top_n features (all when None) are fitted and returned. Every ranking
    method's own top_n selection is also fitted, and those per-split
    metrics land in method_metrics beside the returned "ensemble" fit.
    X must carry the same features that produced the ranking result,
    though the rows may differ. valid and test are held-out (X, y) pairs
    with those same features, never trained on: use valid metrics to
    choose between selections and settings, and quote test for the final
    result. standardize and entropy behave as in fit_convex.
    """
    if entropy < 0.0:
        raise ValueError(f"entropy must be >= 0, got {entropy}.")
    X_arr, feature_names = _convert_features(X, "float64")  # (n, p)
    if feature_names != result.feature_names:
        raise ValueError(
            "X does not match the ranked features; pass a matrix with the "
            "same columns that produced this RankingResult."
        )
    y_arr, classes = _binary_target(y, result.task, X_arr.shape[0])  # (n,)

    evals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, pair in (("valid", valid), ("test", test)):
        if pair is None:
            continue
        X_eval, y_eval = _check_eval_pair(pair, split_name)
        X_eval_arr, eval_names = _convert_features(X_eval, "float64")  # (n_eval, p)
        if eval_names != result.feature_names:
            raise ValueError(
                f"{split_name} X does not match the ranked features; pass the "
                "same columns that produced this RankingResult."
            )
        y_eval_arr = _encode_eval_target(
            y_eval, result.task, classes, X_eval_arr.shape[0], split_name
        )  # (n_eval,)
        evals[split_name] = (X_eval_arr, y_eval_arr)

    if top_n is None:
        top_n = result.n_features
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}.")
    if top_n > result.n_features:
        logger.info(
            "top_n=%d exceeds the %d ranked features; fitting all of them.",
            top_n, result.n_features,
        )
        top_n = result.n_features

    consensus = voting(result, weights=weights, method=vote_method)
    column_index = {name: i for i, name in enumerate(feature_names)}

    def fit_selection(selected: tuple[str, ...]) -> ConvexFit:
        indices = [column_index[name] for name in selected]
        fit = _fit(
            X_arr[:, indices], y_arr, selected, result.task, standardize, entropy
        )
        if not evals:
            return fit
        metrics = dict(fit.metrics)
        for split_name, (X_eval_arr, y_eval_arr) in evals.items():
            scores = fit.predict(X_eval_arr[:, indices])  # (n_eval,)
            metrics[split_name] = _metric(result.task, y_eval_arr, scores)
        return replace(fit, metrics=metrics)

    ensemble = fit_selection(tuple(consensus["feature"].head(top_n)))
    if top_n == result.n_features:
        # every selection is the same full feature set, so every fit
        # reaches the same optimum; skip the redundant solves
        method_metrics = {method: dict(ensemble.metrics) for method in result.methods}
    else:
        method_metrics = {
            method: fit_selection(tuple(table["feature"].head(top_n))).metrics
            for method, table in result.rankings.items()
        }
    method_metrics["ensemble"] = dict(ensemble.metrics)
    for split_name in ensemble.metrics:
        logger.info(
            "Convex fit %s on %s by selection: %s.",
            ensemble.metric_name,
            split_name,
            ", ".join(
                f"{name}={values[split_name]:.4f}"
                for name, values in method_metrics.items()
            ),
        )
    return replace(ensemble, method_metrics=method_metrics)
