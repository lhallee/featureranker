# Performance and parallelism

## The n_jobs model

`feature_ranking(n_jobs=...)` is a total core budget, not a worker count.
Methods run sequentially; each method spends the whole budget internally:

| method | where the budget goes |
|---|---|
| rf, xg | halving search processes across candidates (estimators stay single-threaded during the search), then one final fit with `n_jobs=budget` |
| l1 classification | parallel wave of single-threaded logistic fits |
| l1 regression | multithreaded BLAS in the Gram pass (exact route) |
| mi | `n_jobs` inside the kNN mutual information estimator |
| f_test | negligible cost, single-threaded |

v2 instead ran the methods themselves in parallel worker processes. That
demoted each method's internal parallelism to sequential (joblib forbids
nested process pools) and pinned XGBoost's OpenMP threads, so the documented
`n_jobs=-1` configuration could run slower than the default. The v3 layout
removes that trap structurally: there is never a process pool inside a
process pool.

`n_jobs=-1` (default) resolves to all cores; positive integers cap the
budget; 0 and other negatives raise `ValueError`.

## One conversion, shared arrays

`feature_ranking` converts the DataFrame to one C-order numpy array up front
(float32 by default, `dtype="float64"` for the escape hatch) and hands that
single array to every method. Worker processes receive it through joblib's
automatic memmapping, not per-task pickling. Each L1 route builds at most one
standardized copy; the exact lasso route builds none at all, standardizing
through Gram algebra instead, which is what keeps tall data (millions of
rows, ~1e3 features) inside memory: the dominant allocation is the shared
float32 array itself.

## Adaptive strategy selection

- L1 regression picks the exact LARS path when `p <= 4096` and
  `n * p^2 <= 4e12` (a Gram build bounded in memory and BLAS time), else the
  dense coordinate-descent grid. Override with
  `options={"l1": {"strategy": "exact"}}` or `"grid"`.
- L1 classification picks liblinear (fast, binary, up to 50,000 rows) or
  saga (multiclass, large n). Override with `options={"l1": {"solver": ...}}`.
- Tree searches subsample rows through successive halving rungs sized to the
  dataset; tiny datasets fall back to a plain randomized search.
- Mutual information subsamples above 100,000 rows (seeded, logged;
  `max_samples=None` disables).

## Determinism

Every stochastic component is seeded from the single `random_state`
argument (default 42): tree searches and their CV shuffles, both final
fits, the mutual information estimator and its subsample, and both L1
solvers. Two calls with identical inputs return results for which
`first.equals(second)` is True, at any `n_jobs`; the test suite enforces
this, including `n_jobs=1` against `n_jobs=4`.

## Costs worth knowing

- The tree searches are the most expensive methods end to end; with
  27 candidates and factor-3 halving the search costs about 12 full-data
  fit equivalents per model at cv=3, spread over cheap small rungs.
- The L1 classification waves typically spend 40 to 80 single-threaded fits,
  all parallel, at `tol=1e-4`. v2 spent 50 sequential saga fits at
  `tol=1e-6` with a 10^6 iteration cap.
- The grid lasso route is a single-core coordinate descent chain (warm
  starts are inherently sequential); it is the one stage a core budget does
  not accelerate.
- GPU XGBoost (`estimator_kwargs={"device": "cuda"}`) pays off at large n;
  each halving rung refits with host-to-device transfer, so small datasets
  are faster on CPU.

## Benchmarks

Run `python bench/bench_rankers.py --scenario small-cls` (also `tall-cls`,
`tall-reg`, `wide-cls`, `wide-reg`; `--scale` grows the shapes) to time the
methods on synthetic data with known informative features.

Measured 2026-08-14 on a 16-core AMD Zen 2 desktop (Windows 10, Python
3.11.9, scikit-learn 1.8.0, xgboost 3.2.0, identical pinned dependencies in
both environments). v2 rows use its best mode (`n_jobs=1`; its `n_jobs=-1`
was slower, 5.6s vs 5.1s for the small ensemble, because of the nested
parallelism it triggered). v3 rows use `n_jobs=-1`. Seconds per method:

| scenario | method | v2.0.0 | v3.0.0 | speedup |
|---|---|---|---|---|
| 20,000 x 500 | l1 | 236.6 | 8.3 | 28x |
| 20,000 x 500 | mi | 36.9 | 4.0 | 9.2x |
| 20,000 x 500 | rf + xg | not finished in 600 | 167.0 | > 3.6x |
| 20,000 x 500 | full ensemble | not finished in 600 | 171.8 | > 3.5x |
| 2,000 x 5,000 | l1 | not finished in 300 | 9.0 | > 33x |
| 2,000 x 5,000 | mi | 30.6 | 3.5 | 8.7x |
| 2,000 x 5,000 | rf | 347.0 | 225.7 | 1.5x |
| 2,000 x 5,000 | xg | 180.7 | 138.3 | 1.3x |
| 500 x 50 | full ensemble | 5.1 | 3.4 | 1.5x |

Two caveats the table earns honestly: the v3 tree searches explore 27
hyperparameter candidates against v2's 5, so their speedups understate the
per-candidate gain; and on tiny data with `n_jobs=1` the v3 rf search is
slower than v2's (13.8s vs 3.8s on 500 x 50) because that exploration has
nothing to parallelize against, which is the price of the better search at
sizes where either finishes in seconds.
