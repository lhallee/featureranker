"""Entry point orchestrating the ranking methods over one dataset."""

import logging
import time

import joblib
import numpy as np
import pandas as pd

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from typing import Literal

from sklearn.preprocessing import LabelEncoder

from .lasso import LassoOptions, rank_lasso
from .logistic import LogisticL1Options, rank_logistic_l1
from .result import RankingResult, make_table
from .trees import TreeSearchOptions, rank_random_forest, rank_xgboost
from .univariate import MutualInfoOptions, rank_f_test, rank_mutual_info

logger = logging.getLogger(__name__)

METHODS: tuple[str, ...] = ("rf", "xg", "mi", "f_test", "l1")

RankerFn = Callable[..., tuple[np.ndarray, dict[str, object]]]

# task-independent methods: key -> (ranker, options dataclass or None)
_REGISTRY: dict[str, tuple[RankerFn, type | None]] = {
    "rf": (rank_random_forest, TreeSearchOptions),
    "xg": (rank_xgboost, TreeSearchOptions),
    "mi": (rank_mutual_info, MutualInfoOptions),
    "f_test": (rank_f_test, None),
}


def _method_spec(method: str, task: str) -> tuple[RankerFn, type | None]:
    """Resolve the ranker and options type; l1 depends on the task."""
    if method == "l1":
        if task == "regression":
            return rank_lasso, LassoOptions
        return rank_logistic_l1, LogisticL1Options
    return _REGISTRY[method]


def _resolve_budget(n_jobs: int) -> int:
    """Map the n_jobs argument onto a concrete core budget."""
    cores = joblib.cpu_count()
    if n_jobs == -1 or n_jobs is None:
        return cores
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be -1 or a positive integer, got {n_jobs!r}.")
    return min(n_jobs, cores)


def _resolve_options(method: str, supplied: object, options_type: type | None) -> object:
    """Build the method's options dataclass from a dict, an instance, or None."""
    if supplied is None:
        return options_type() if options_type is not None else None
    if options_type is None:
        raise ValueError(f"Method {method!r} takes no options.")
    if isinstance(supplied, options_type):
        return supplied
    if isinstance(supplied, Mapping):
        allowed = {field.name for field in fields(options_type)}
        unknown = set(supplied) - allowed
        if unknown:
            raise ValueError(
                f"Unknown options for {method!r}: {sorted(unknown)}. "
                f"Valid: {sorted(allowed)}."
            )
        return options_type(**supplied)
    raise TypeError(
        f"Options for {method!r} must be a dict or {options_type.__name__}, "
        f"got {type(supplied).__name__}."
    )


def generated_feature_names(n_features: int) -> tuple[str, ...]:
    """Stable IDs for unnamed features: f000..f999 style, zero-padded."""
    width = max(len(str(n_features - 1)), 1)
    return tuple(f"f{i:0{width}d}" for i in range(n_features))


def _convert_features(
    X: pd.DataFrame | np.ndarray, dtype: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert the feature matrix to one shared numeric array plus names.

    A DataFrame keeps its column names; a bare 2D array (an embedding matrix,
    pooled hidden states, any engineered dense vector) gets generated IDs.
    """
    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"A numpy X must be 2D (n_samples, n_features), got ndim={X.ndim}.")
        feature_names = generated_feature_names(X.shape[1])
        source = X
    elif isinstance(X, pd.DataFrame):
        feature_names = tuple(str(name) for name in X.columns)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X has duplicate column names.")
        source = None
    else:
        raise TypeError(
            f"X must be a pandas DataFrame or a 2D numpy array, got {type(X).__name__}."
        )
    if X.shape[1] == 0 or X.shape[0] < 2:
        raise ValueError(f"X needs at least 2 rows and 1 column, got shape {X.shape}.")

    try:
        if source is None:
            X_arr = np.ascontiguousarray(X.to_numpy(dtype=np.dtype(dtype)))  # (n, p)
        else:
            X_arr = np.ascontiguousarray(source, dtype=np.dtype(dtype))  # (n, p)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"X contains non-numeric data; encode it first with get_data(): {error}"
        ) from error
    if not np.isfinite(X_arr).all():
        raise ValueError(
            "X contains NaN or infinite values; clean it first with get_data() "
            "(view_data() shows the affected columns)."
        )
    return X_arr, feature_names


def _convert_target(
    y: pd.Series | np.ndarray, task: str, n_samples: int
) -> tuple[np.ndarray, tuple[object, ...] | None]:
    """Convert the target to the array the rankers consume."""
    y_arr = np.asarray(y).ravel()  # (n,)
    if y_arr.shape[0] != n_samples:
        raise ValueError(f"X has {n_samples} rows but y has {y_arr.shape[0]}.")
    if pd.isna(y_arr).any():
        raise ValueError("y contains missing values; drop or impute them first.")

    if task == "classification":
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_arr).astype(np.int32)  # (n,)
        classes = tuple(encoder.classes_.tolist())
        if len(classes) < 2:
            raise ValueError("Classification needs at least 2 classes in y.")
        return y_encoded, classes

    y_float = y_arr.astype(np.float64)  # (n,)
    if not np.isfinite(y_float).all():
        raise ValueError("y contains infinite values.")
    if np.all(y_float == y_float[0]):
        raise ValueError("y is constant; regression needs a varying target.")
    return y_float, None


def feature_ranking(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    methods: Sequence[str] | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    dtype: Literal["float32", "float64"] = "float32",
    options: Mapping[str, object] | None = None,
) -> RankingResult:
    """Rank features with an ensemble of methods and return a RankingResult.

    X is any numeric feature matrix: a DataFrame keeps its column names, and
    a bare 2D numpy array (embeddings, pooled hidden states) gets generated
    IDs like f0000. Methods run sequentially; each consumes the full n_jobs
    core budget internally, which avoids nested-parallelism oversubscription.
    options maps a method key to an option dict or its options dataclass,
    for example options={"mi": {"max_samples": None}}.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task {task!r}. Valid: 'classification', 'regression'.")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"Unknown dtype {dtype!r}. Valid: 'float32', 'float64'.")

    chosen = tuple(methods) if methods is not None else METHODS
    if not chosen:
        raise ValueError("methods is empty; pick from " + ", ".join(METHODS) + ".")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"methods lists a method more than once: {chosen}.")
    unknown_methods = set(chosen) - set(METHODS)
    if unknown_methods:
        raise ValueError(
            f"Unknown methods {sorted(unknown_methods)}. Valid: {METHODS}."
        )

    method_options = dict(options) if options is not None else {}
    stray = set(method_options) - set(chosen)
    if stray:
        raise ValueError(
            f"Options given for methods that will not run: {sorted(stray)}."
        )

    budget = _resolve_budget(n_jobs)
    X_arr, feature_names = _convert_features(X, dtype)  # (n, p)
    y_arr, classes = _convert_target(y, task, X_arr.shape[0])  # (n,)

    rankings: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for method in chosen:
        ranker, options_type = _method_spec(method, task)
        resolved = _resolve_options(method, method_options.get(method), options_type)
        started = time.perf_counter()
        scores, method_diagnostics = ranker(
            X_arr, y_arr, task=task, budget=budget,
            random_state=random_state, options=resolved,
        )  # scores: (p,)
        elapsed = time.perf_counter() - started
        method_diagnostics["seconds"] = round(elapsed, 3)
        rankings[method] = make_table(feature_names, scores)
        diagnostics[method] = method_diagnostics
        logger.info("Method %s finished in %.2f s.", method, elapsed)

    from . import __version__

    return RankingResult(
        task=task,
        feature_names=feature_names,
        n_samples=X_arr.shape[0],
        n_features=X_arr.shape[1],
        rankings=rankings,
        diagnostics=diagnostics,
        classes=classes,
        random_state=random_state,
        version=__version__,
    )
