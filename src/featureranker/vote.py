"""Weighted rank aggregation across ranking methods."""

import numpy as np
import pandas as pd

from collections.abc import Mapping
from typing import Literal

from scipy.stats import rankdata

from .result import RankingResult, make_table

VOTE_METHODS: tuple[str, ...] = ("reciprocal_rank", "borda", "exponential")


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
    weights: Mapping[str, float] | None = None,
    method: Literal["reciprocal_rank", "borda", "exponential"] = "reciprocal_rank",
) -> pd.DataFrame:
    """Aggregate per-method rankings into one table of weighted vote scores.

    Tied scores within a method receive their average rank before points are
    assigned, so exact ties contribute identically. Weights are keyed by method
    name; missing keys default to 1.0 and unknown keys raise.

    Returns a table with columns ["feature", "score"], best first.
    """
    rankings = result.rankings if isinstance(result, RankingResult) else dict(result)
    if not rankings:
        raise ValueError("There are no rankings to aggregate.")
    if method not in VOTE_METHODS:
        raise ValueError(f"Unknown voting method {method!r}. Valid: {VOTE_METHODS}.")

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
