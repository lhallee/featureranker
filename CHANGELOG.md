# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2026-08-14

Breaking release: the ranking engine, public API, and packaging were rebuilt.
See [docs/migration_v2_to_v3.md](docs/migration_v2_to_v3.md) for upgrade
steps.

### Fixed

- L1 classification ranked features by the largest C with a nonzero
  coefficient, which degenerates to ties at the top of the grid; it now
  ranks by the entry point (smallest C), the intended statistic. Rankings
  genuinely change.
- L1 classification crashed on 3+ classes; multiclass now works through the
  any-class entry mask.
- Results were nondeterministic (unseeded saga solver, unstable tie sorts);
  every stochastic component is now seeded and ties order deterministically,
  so identical calls return identical results at any n_jobs.
- `get_data` skipped pandas `category` and datetime columns during encoding
  and returned string classification targets unencoded.
- Plots saved the current figure instead of the target axes' figure.
- `plot_rankings` drew synthetic positional scores; it now draws real ranks.
- `requires-python >= 3.9` was declared but the code needed 3.10+.

### Changed

- `feature_ranking` returns a frozen `RankingResult` (rankings keyed by
  method, rank/score matrices, diagnostics, save/load) instead of a list of
  tuples; `choices` renamed to `methods`; per-method `options` replace
  silent `**kwargs`.
- `voting` takes dict weights keyed by method name and accepts a
  `RankingResult`; the exponential scheme is normalized to span 1 to
  exp(-1).
- `n_jobs` became a total core budget spent inside each method; methods run
  sequentially, ending v2's nested-parallelism slowdown where `n_jobs=-1`
  could run slower than the default.
- Tree models tune with successive halving (27 candidates on subsampled
  rungs, balanced accuracy / RMSE scoring, tightened spaces) and fit the
  final model exactly once with the full core budget.
- L1 regression computes exact path breakpoints via a Gram-based LARS when
  affordable and a dense warm-started coordinate-descent grid otherwise.
- L1 classification brackets entry points with parallel coarse-to-fine C
  waves instead of 50 sequential saga fits.
- Mutual information uses n_jobs and subsamples above 100,000 rows (seeded,
  logged, `max_samples=None` for exact).
- Input validation raises `ValueError`/`TypeError` instead of assertions.
- Plots redesigned: consensus-ordered dot plot, new rank heatmap, emphasis
  vote bars, recessive chrome, colorblind-validated fixed method colors.
- License changed from CC-BY-NC-SA-4.0 to MIT.
- Floors raised: Python >= 3.11, scikit-learn >= 1.8, xgboost >= 2.0,
  numpy >= 1.26, pandas >= 2.1, scipy >= 1.11, matplotlib >= 3.8,
  seaborn >= 0.13, joblib >= 1.3.

### Added

- `RankingResult` with `rank_matrix`, `score_matrix`, `equals`, `save`,
  `load`, and per-method diagnostics.
- Typed per-method options: `LassoOptions`, `LogisticL1Options`,
  `TreeSearchOptions`, `MutualInfoOptions`.
- `plot_rank_heatmap`.
- GPU XGBoost passthrough via
  `options={"xg": {"estimator_kwargs": {"device": "cuda"}}}`.
- `py.typed` (PEP 561), single-source versioning through hatchling, test CI
  across Python 3.11-3.13 on Linux and Windows, `docs/` reference set,
  benchmark script `bench/bench_rankers.py`.

### Removed

- The clustering module and its five exports.
- `make_ranking`, `l1_regression_ranking`, `l1_classification_ranking`,
  `hyper_param_search`, `VALID_CHOICES`, and the `save=`/`save_path=`
  parameters on `feature_ranking`/`voting`.
- `requirements.txt` (pyproject is the single dependency source), the stale
  root `test.py`, and the `documentation/` folder (replaced by `docs/`).

## [2.0.0] - 2026 (tag `2.0.0`)

v2 rewrite: five-method ensemble with joblib dispatch, pytest suite,
hatchling packaging.
