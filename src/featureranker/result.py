"""Ranking result container and the shared ranking-table schema."""

import warnings

import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from scipy.stats import rankdata


def make_table(feature_names: tuple[str, ...], scores: np.ndarray) -> pd.DataFrame:
    """Build the uniform ranking table with columns ["feature", "score"], best first.

    Sorting is stable with the feature name as tiebreak, so tied scores keep a
    deterministic order across runs and platforms.
    """
    scores = np.asarray(scores, dtype=np.float64)  # (p,)
    if len(feature_names) != scores.shape[0]:
        raise ValueError(
            f"Got {len(feature_names)} feature names but {scores.shape[0]} scores."
        )
    table = pd.DataFrame({"feature": feature_names, "score": scores})
    table = table.sort_values(
        ["score", "feature"], ascending=[False, True], kind="mergesort"
    )
    return table.reset_index(drop=True)


@dataclass(frozen=True, eq=False)
class RankingResult:
    """Feature rankings produced by one feature_ranking run.

    Each table in rankings has columns ["feature", "score"], sorted best first,
    with scores oriented so that higher is always better. Diagnostics hold
    method internals (fit counts, search parameters, raw entry thresholds) and
    are informational: they are excluded from equals().
    """

    task: str
    feature_names: tuple[str, ...]
    n_samples: int
    n_features: int
    rankings: dict[str, pd.DataFrame]
    diagnostics: dict[str, dict[str, object]]
    classes: tuple[object, ...] | None
    random_state: int
    version: str

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.rankings)

    def score_matrix(self) -> pd.DataFrame:
        """Raw scores per feature and method, rows in feature_names order."""
        columns = {
            method: table.set_index("feature")["score"]
            for method, table in self.rankings.items()
        }
        S = pd.DataFrame(columns)  # (p, m)
        return S.reindex(list(self.feature_names))

    def rank_matrix(self) -> pd.DataFrame:
        """Average ranks per feature and method, 1 = best; ties share ranks."""
        S = self.score_matrix().to_numpy(dtype=np.float64)  # (p, m)
        R = rankdata(-S, method="average", axis=0)  # (p, m)
        return pd.DataFrame(
            R, index=list(self.feature_names), columns=list(self.rankings)
        )

    def equals(self, other: "RankingResult") -> bool:
        """Exact equality of the scientific payload; diagnostics are excluded."""
        if not isinstance(other, RankingResult):
            return False
        same_metadata = (
            self.task == other.task
            and self.feature_names == other.feature_names
            and self.n_samples == other.n_samples
            and self.n_features == other.n_features
            and self.classes == other.classes
            and self.random_state == other.random_state
            and self.version == other.version
        )
        if not same_metadata or self.methods != other.methods:
            return False
        return all(
            self.rankings[method].equals(other.rankings[method])
            for method in self.rankings
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "RankingResult":
        """Load a saved result; warn when the package version differs."""
        # deserialization boundary: any failure mode maps onto the documented
        # ValueError contract
        try:
            loaded = joblib.load(path)
        except Exception as error:
            raise ValueError(f"Could not load a RankingResult from {path}: {error}") from error
        if not isinstance(loaded, RankingResult):
            raise ValueError(f"{path} does not contain a RankingResult.")

        from . import __version__

        if loaded.version != __version__:
            warnings.warn(
                f"Result was saved by featureranker {loaded.version}, "
                f"loading under {__version__}.",
                stacklevel=2,
            )
        return loaded

    def __repr__(self) -> str:
        return (
            f"RankingResult(task={self.task!r}, n_samples={self.n_samples}, "
            f"n_features={self.n_features}, methods={self.methods!r})"
        )
