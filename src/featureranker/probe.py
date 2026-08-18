"""Probe evaluation: how predictive each method's top features are.

One shared protocol makes the methods comparable: for a ladder of top-k
cuts through each method's ranking, a standardized linear probe (logistic
regression or ridge) is cross-validated and the scores averaged over the
ladder. The averaged score is normalized into a skill in [0, 1] (0 = chance
or worse), which voting(weights="auto") uses as that method's vote weight.
"""

import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROBE_KS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
PROBE_FOLDS = 3
PROBE_MAX_SAMPLES = 10_000


def _probe_setup(y: np.ndarray, task: str, random_state: int):
    """Choose the probe model, CV splitter, and scoring for the task."""
    # y: (n,)
    if task == "classification":
        counts = np.bincount(y.astype(np.int64))  # (c,)
        folds = min(PROBE_FOLDS, int(counts[counts > 0].min()))
        if folds < 2:
            return None, None, ""
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
        return model, splitter, "balanced_accuracy"
    if y.shape[0] < 2 * PROBE_FOLDS:
        return None, None, ""
    model = make_pipeline(StandardScaler(), Ridge())
    splitter = KFold(PROBE_FOLDS, shuffle=True, random_state=random_state)
    return model, splitter, "r2"


def _skill(score: float, task: str, n_classes: int) -> float:
    """Normalize a probe score into [0, 1] skill above chance.

    Balanced accuracy has chance level 1/c, so skill rescales the span from
    chance to perfect; R2 is already 0 at the predict-the-mean baseline.
    """
    if task == "classification":
        chance = 1.0 / n_classes
        return max((score - chance) / (1.0 - chance), 0.0)
    return max(score, 0.0)


def probe_rankings(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    rankings: dict[str, pd.DataFrame],
    feature_names: tuple[str, ...],
    budget: int,
    random_state: int,
) -> dict[str, dict[str, object]]:
    """Probe every method's ranking and return per-method reports.

    Each report holds the metric name, the cross-validated score at each
    top-k cut, their mean as "score", and the normalized "skill". Returns
    an empty dict when the data cannot support cross-validation.
    """
    # X: (n, p); y: (n,)
    model, splitter, scoring = _probe_setup(y, task, random_state)
    if model is None:
        logger.warning("Too few samples per class for probe cross-validation; skipping.")
        return {}

    if X.shape[0] > PROBE_MAX_SAMPLES:
        rng = np.random.default_rng(random_state)
        rows = np.sort(rng.choice(X.shape[0], size=PROBE_MAX_SAMPLES, replace=False))  # (m,)
        X, y = X[rows], y[rows]
        logger.info("Probing on a seeded subsample of %d rows.", PROBE_MAX_SAMPLES)

    n_classes = int(y.max()) + 1 if task == "classification" else 0
    column_index = {name: i for i, name in enumerate(feature_names)}
    ks = [k for k in PROBE_KS if k <= len(feature_names)]

    reports: dict[str, dict[str, object]] = {}
    for method, table in rankings.items():
        ranked_columns = [column_index[name] for name in table["feature"]]  # (p,)
        by_k: dict[int, float] = {}
        for k in ks:
            X_top = X[:, ranked_columns[:k]]  # (n, k)
            folds = cross_val_score(
                model, X_top, y, cv=splitter, scoring=scoring, n_jobs=budget
            )  # (folds,)
            by_k[k] = float(folds.mean())
        score = float(np.mean(list(by_k.values())))
        reports[method] = {
            "metric": scoring,
            "by_k": by_k,
            "score": score,
            "skill": _skill(score, task, n_classes),
        }
    logger.info(
        "Probe %s by method: %s.",
        scoring,
        ", ".join(f"{m}={r['score']:.4f}" for m, r in reports.items()),
    )
    return reports
