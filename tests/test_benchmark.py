"""Slow wall-clock smoke benchmarks; run with pytest -m slow."""

import time

import numpy as np
import pandas as pd
import pytest

from featureranker.ranking import feature_ranking


def _shifted(n: int, p: int, k: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=n)  # (n,)
    X = rng.normal(size=(n, p))  # (n, p)
    X[:, :k] += 2.0 * y[:, None]
    return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(p)]), pd.Series(y)


@pytest.mark.slow
def test_full_ensemble_wall_time():
    X, y = _shifted(500, 50, 10)
    started = time.perf_counter()
    result = feature_ranking(X, y)
    seconds = time.perf_counter() - started
    assert result.methods == ("rf", "xg", "mi", "f_test", "l1")
    assert seconds < 180, f"full ensemble took {seconds:.1f} s on 500x50"


@pytest.mark.slow
def test_wide_l1_wall_time():
    X, y = _shifted(200, 1000, 20)
    started = time.perf_counter()
    result = feature_ranking(X, y, methods=["l1"])
    seconds = time.perf_counter() - started
    assert set(result.rankings["l1"]["feature"]) == set(X.columns)
    assert seconds < 120, f"wide L1 took {seconds:.1f} s on 200x1000"
