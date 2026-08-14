"""L1 classification ranking: the entry C of each feature, found in parallel waves.

A feature's entry C is the smallest inverse-regularization strength at which
any of its class coefficients is nonzero; features that enter under stronger
regularization (smaller C) matter more, so the score is 1 / entry_C. Instead
of one tight sequential path, independent single-threaded fits fan out across
cores in waves: a coarse log-grid brackets every feature's entry at once, then
refinement waves subdivide only the still-open brackets.
"""

import logging

import numpy as np

from dataclasses import dataclass
from typing import Literal

from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

logger = logging.getLogger(__name__)

_COEF_ATOL = 1e-10


@dataclass(frozen=True)
class LogisticL1Options:
    """Options for the L1 classification ranker.

    The coarse wave samples coarse_size points over `decades` decades above
    l1_min_c; features still absent get one extension wave reaching
    max_extra_decades further. Brackets refine until hi/lo <= 1 + rtol or the
    wave/fit budget runs out. liblinear is used for binary problems up to
    liblinear_max_n rows; saga handles multiclass and larger data.
    """

    solver: Literal["auto", "liblinear", "saga"] = "auto"
    coarse_size: int = 16
    decades: float = 3.0
    max_extra_decades: float = 2.0
    rtol: float = 0.15
    max_waves: int = 8
    max_fits: int = 128
    tol: float = 1e-4
    max_iter: int = 5000
    liblinear_max_n: int = 50_000


def _fit_entry_mask(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    solver: str,
    tol: float,
    max_iter: int,
    random_state: int,
) -> np.ndarray:
    """One single-threaded L1 logistic fit; True where a feature is active."""
    # X: (n, p) standardized; y: (n,)
    clf = LogisticRegression(
        C=C,
        l1_ratio=1.0,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        # liblinear penalizes the intercept; a large scaling approximates the
        # unpenalized intercept that saga fits natively
        intercept_scaling=100.0 if solver == "liblinear" else 1.0,
    )
    clf.fit(X, y)
    W = clf.coef_  # (c', p) with c' = 1 for binary
    return (np.abs(W) > _COEF_ATOL).any(axis=0)  # (p,)


def _standardized_copy(X: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """One standardized C-order copy of the shared array."""
    mu = X.mean(axis=0, dtype=np.float64)  # (p,)
    sd = X.std(axis=0, dtype=np.float64)  # (p,)
    sd[sd == 0.0] = 1.0
    Xs = np.empty(X.shape, dtype=dtype)  # (n, p)
    np.subtract(X, mu.astype(dtype), out=Xs)
    Xs /= sd.astype(dtype)
    return Xs


def _refinement_candidates(
    lo: np.ndarray, hi: np.ndarray, rtol: float, cap: int
) -> list[float]:
    """Geometric midpoints of open brackets, deduped, widest brackets first."""
    # lo, hi: (p,)
    open_brackets = (lo > 0.0) & np.isfinite(hi) & (hi > lo * (1.0 + rtol))  # (p,)
    if not open_brackets.any():
        return []
    midpoints = np.sqrt(lo[open_brackets] * hi[open_brackets])  # (n_open,)
    widths = hi[open_brackets] / lo[open_brackets]  # (n_open,)
    order = np.argsort(-widths, kind="stable")
    seen: set[float] = set()
    candidates: list[float] = []
    for C, _ in zip(midpoints[order], widths[order]):
        key = round(float(np.log10(C)), 3)
        if key not in seen:
            seen.add(key)
            candidates.append(float(C))
        if len(candidates) == cap:
            break
    return sorted(candidates)


def rank_logistic_l1(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budget: int,
    random_state: int,
    options: LogisticL1Options,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score features by 1 / entry_C on the L1 logistic regularization path."""
    # X: (n, p); y: (n,) encoded 0..k-1
    n, p = X.shape
    n_classes = int(y.max()) + 1

    solver = options.solver
    if solver == "auto":
        solver = (
            "liblinear"
            if n_classes == 2 and n <= options.liblinear_max_n
            else "saga"
        )
    if solver == "liblinear" and n_classes > 2:
        raise ValueError(
            "liblinear handles binary targets only; use options="
            '{"l1": {"solver": "saga"}} for multiclass.'
        )

    # liblinear solves in float64; saga accepts the shared dtype directly
    Xs = _standardized_copy(X, np.float64 if solver == "liblinear" else X.dtype)  # (n, p)
    c_min = float(
        l1_min_c(
            Xs,
            y,
            loss="log",
            fit_intercept=True,
            intercept_scaling=100.0 if solver == "liblinear" else 1.0,
        )
    )

    lo = np.zeros(p)  # (p,) largest C seen all-zero below the entry
    hi = np.full(p, np.inf)  # (p,) smallest C seen nonzero
    n_fits = 0
    n_waves = 0

    def absorb(wave_Cs: list[float], masks: list[np.ndarray]) -> None:
        nonlocal lo, hi
        for C, mask in sorted(zip(wave_Cs, masks), key=lambda pair: pair[0]):
            hi = np.where(mask, np.minimum(hi, C), hi)
            below_entry = (~mask) & (C < hi)  # (p,) ignore re-exits above entry
            lo = np.where(below_entry, np.maximum(lo, C), lo)

    with Parallel(n_jobs=min(budget, options.max_fits), backend="loky") as parallel:

        def run_wave(wave_Cs: list[float]) -> None:
            nonlocal n_fits, n_waves
            masks = parallel(
                delayed(_fit_entry_mask)(
                    Xs, y, C, solver, options.tol, options.max_iter, random_state
                )
                for C in wave_Cs
            )
            absorb(wave_Cs, masks)
            n_fits += len(wave_Cs)
            n_waves += 1

        coarse = c_min * np.logspace(0.0, options.decades, options.coarse_size)  # (coarse_size,)
        run_wave([float(C) for C in coarse])

        if np.isinf(hi).any():
            extension = c_min * np.logspace(
                options.decades, options.decades + options.max_extra_decades, 8
            )  # (8,)
            run_wave([float(C) for C in extension[1:]])

        while n_waves < options.max_waves and n_fits < options.max_fits:
            cap = min(max(budget, 16), options.max_fits - n_fits)
            candidates = _refinement_candidates(lo, hi, options.rtol, cap)
            if not candidates:
                break
            run_wave(candidates)

    entry_C = hi  # (p,)
    scores = np.where(np.isfinite(entry_C), 1.0 / entry_C, 0.0)  # (p,)
    still_open = (lo > 0.0) & np.isfinite(hi) & (hi > lo * (1.0 + options.rtol))
    diagnostics = {
        "solver": solver,
        "c_min": c_min,
        "n_fits": n_fits,
        "n_waves": n_waves,
        "n_never_entered": int(np.isinf(entry_C).sum()),
        "n_unresolved": int(still_open.sum()),
        "entry_C": entry_C.copy(),
    }
    return scores, diagnostics
