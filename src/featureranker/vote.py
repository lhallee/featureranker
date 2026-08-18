"""Weighted rank aggregation across ranking methods."""

import logging

import numpy as np
import pandas as pd

from collections.abc import Mapping
from typing import Literal

from scipy.stats import rankdata

from .result import RankingResult, make_table

logger = logging.getLogger(__name__)

VOTE_METHODS: tuple[str, ...] = ("reciprocal_rank", "borda", "exponential")


def _auto_weights(result: RankingResult | Mapping[str, pd.DataFrame]) -> dict[str, float]:
    """Vote weights from probe skill: more predictive methods vote harder."""
    if not isinstance(result, RankingResult):
        raise ValueError(
            "weights='auto' needs a RankingResult; a plain rankings mapping "
            "carries no probe reports."
        )
    skills: dict[str, float] = {}
    for method in result.methods:
        report = result.diagnostics.get(method, {}).get("probe")
        if report is None:
            raise ValueError(
                f"Method {method!r} has no probe report; rerun feature_ranking "
                "with probe=True to use weights='auto'."
            )
        skills[method] = float(report["skill"])
    if all(skill == 0.0 for skill in skills.values()):
        logger.warning("Every method probed at chance level; using equal weights.")
        return {method: 1.0 for method in skills}
    logger.info(
        "Auto vote weights from probe skill: %s.",
        ", ".join(f"{method}={skill:.4f}" for method, skill in skills.items()),
    )
    return skills


def _rank_points(ranks: np.ndarray, n_features: int, method: str) -> np.ndarray:
    """Convert average ranks (1 = best) into vote points for one method."""
    # ranks: (p,)
    if method == "reciprocal_rank":
        return 1.0 / ranks  # (p,)
    if method == "borda":
        return n_features - ranks  # (p,)
    # exponential: best rank scores 1.0, worst scores exp(-1); flat for p == 1
    return np.exp(-(ranks - 1.0) / max(n_features - 1, 1))  # (p,)


def voting(
    result: RankingResult | Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | Literal["auto"] | None = None,
    method: Literal["reciprocal_rank", "borda", "exponential"] = "reciprocal_rank",
) -> pd.DataFrame:
    """Aggregate per-method rankings into one table of weighted vote scores.

    Tied scores within a method receive their average rank before points are
    assigned, so exact ties contribute identically. Weights are keyed by method
    name; missing keys default to 1.0 and unknown keys raise. weights="auto"
    weights each method by its probe skill from feature_ranking(probe=True),
    so more predictive methods vote harder.

    Returns a table with columns ["feature", "score"], best first.
    """
    rankings = result.rankings if isinstance(result, RankingResult) else dict(result)
    if not rankings:
        raise ValueError("There are no rankings to aggregate.")
    if method not in VOTE_METHODS:
        raise ValueError(f"Unknown voting method {method!r}. Valid: {VOTE_METHODS}.")

    if isinstance(weights, str):
        if weights != "auto":
            raise ValueError(f"Unknown weights {weights!r}; pass a mapping or 'auto'.")
        weights = _auto_weights(result)
    weights = dict(weights) if weights is not None else {}
    unknown = set(weights) - set(rankings)
    if unknown:
        raise ValueError(
            f"Weights given for unknown methods {sorted(unknown)}. "
            f"Valid: {sorted(rankings)}."
        )
    for name, weight in weights.items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise TypeError(f"Weight for {name!r} must be a number, got {weight!r}.")

    totals: pd.Series | None = None
    for name, table in rankings.items():
        features = table["feature"].to_numpy()  # (p,)
        if len(np.unique(features)) != len(features):
            raise ValueError(f"Ranking {name!r} lists a feature more than once.")
        scores = table["score"].to_numpy(dtype=np.float64)  # (p,)
        ranks = rankdata(-scores, method="average")  # (p,)
        points = weights.get(name, 1.0) * _rank_points(ranks, len(features), method)  # (p,)
        contribution = pd.Series(points, index=features)  # (p,)
        totals = contribution if totals is None else totals.add(contribution, fill_value=0.0)

    return make_table(tuple(totals.index), totals.to_numpy())
