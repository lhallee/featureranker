"""Random forest and XGBoost ranking: halving search, then one final fit.

The search runs successive halving over subsampled rungs with single-threaded
estimators (process parallelism across candidates), never refits internally,
and the winning parameters get exactly one full-data fit with the whole core
budget. Importances are impurity gain from that final model.
"""

import logging
import math

import numpy as np

from collections.abc import Mapping
from dataclasses import dataclass, field
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

_RF_SPACE = {
    "n_estimators": randint(150, 501),
    "max_depth": [None, 8, 12, 16, 24, 32],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "min_samples_split": randint(2, 16),
    "min_samples_leaf": randint(1, 8),
    "bootstrap": [True],
}
_XGB_SPACE = {
    "n_estimators": randint(150, 501),
    "learning_rate": loguniform(0.02, 0.3),
    "max_depth": randint(3, 9),
    "min_child_weight": loguniform(0.5, 8.0),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma": loguniform(1e-3, 1.0),
    "reg_alpha": loguniform(1e-3, 10.0),
    "reg_lambda": loguniform(0.1, 10.0),
}


@dataclass(frozen=True)
class TreeSearchOptions:
    """Options for the tree-model rankers.

    scoring=None picks balanced accuracy for classification and negative RMSE
    for regression. estimator_kwargs go into both the searched and the final
    estimator, for example {"device": "cuda"} for XGBoost.
    """

    cv: int = 3
    n_candidates: int = 27
    factor: int = 3
    scoring: str | None = None
    estimator_kwargs: Mapping[str, object] = field(default_factory=dict)


def _make_estimator(
    model: str, task: str, random_state: int, n_jobs: int, extra: Mapping[str, object]
):
    if model == "rf":
        forest = (
            RandomForestClassifier if task == "classification" else RandomForestRegressor
        )
        return forest(random_state=random_state, n_jobs=n_jobs, **extra)
    booster = XGBClassifier if task == "classification" else XGBRegressor
    return booster(
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
        **extra,
    )


def _rung_plan(
    n_samples: int, cv: int, n_classes: int, task: str, options: TreeSearchOptions
) -> tuple[int, int, int]:
    """Halving schedule (n_rungs, n_candidates, min_resources) for this dataset."""
    if task == "classification":
        resource_floor = max(2 * cv * n_classes, 32)
    else:
        resource_floor = max(4 * cv, 32)

    if n_samples <= resource_floor:
        return 0, 0, 0
    max_rungs = int(math.log(n_samples / resource_floor) / math.log(options.factor))
    n_rungs = min(3, max_rungs)
    if n_rungs == 0:
        return 0, 0, 0
    n_candidates = min(options.n_candidates, options.factor**n_rungs)
    min_resources = max(resource_floor, n_samples // options.factor**n_rungs)
    return n_rungs, n_candidates, min_resources


def _rank_tree(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budget: int,
    random_state: int,
    options: TreeSearchOptions,
) -> tuple[np.ndarray, dict[str, object]]:
    # X: (n, p); y: (n,)
    n, p = X.shape
    n_classes = int(y.max()) + 1 if task == "classification" else 0

    if task == "classification":
        class_counts = np.bincount(y)  # (k,)
        if class_counts.min() < options.cv:
            raise ValueError(
                f"The smallest class has {class_counts.min()} members but cv="
                f"{options.cv}; reduce cv or gather more samples."
            )
        cv_obj = StratifiedKFold(options.cv, shuffle=True, random_state=random_state)
        scoring = options.scoring or "balanced_accuracy"
    else:
        cv_obj = KFold(options.cv, shuffle=True, random_state=random_state)
        scoring = options.scoring or "neg_root_mean_squared_error"

    space = _RF_SPACE if model == "rf" else _XGB_SPACE
    searched = _make_estimator(model, task, random_state, 1, options.estimator_kwargs)
    n_rungs, n_candidates, min_resources = _rung_plan(n, options.cv, n_classes, task, options)

    if n_rungs == 0:
        search = RandomizedSearchCV(
            searched,
            space,
            n_iter=8,
            cv=cv_obj,
            scoring=scoring,
            refit=False,
            random_state=random_state,
            n_jobs=budget,
        )
        search_kind = "randomized"
    else:
        search = HalvingRandomSearchCV(
            searched,
            space,
            n_candidates=n_candidates,
            factor=options.factor,
            resource="n_samples",
            min_resources=min_resources,
            cv=cv_obj,
            scoring=scoring,
            refit=False,
            random_state=random_state,
            n_jobs=budget,
        )
        search_kind = "halving"
    search.fit(X, y)
    best_params = search.best_params_

    final = _make_estimator(
        model, task, random_state, budget, {**best_params, **options.estimator_kwargs}
    )
    final.fit(X, y)
    scores = final.feature_importances_  # (p,)

    diagnostics = {
        "model": model,
        "search": search_kind,
        "scoring": scoring,
        "best_params": dict(best_params),
        "best_score": float(search.best_score_),
        "n_candidates": n_candidates or 8,
        "min_resources": min_resources,
    }
    return scores.astype(np.float64), diagnostics


def rank_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budget: int,
    random_state: int,
    options: TreeSearchOptions,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score features by random forest impurity importance."""
    return _rank_tree("rf", X, y, task, budget, random_state, options)


def rank_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budget: int,
    random_state: int,
    options: TreeSearchOptions,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score features by XGBoost gain importance."""
    return _rank_tree("xg", X, y, task, budget, random_state, options)
