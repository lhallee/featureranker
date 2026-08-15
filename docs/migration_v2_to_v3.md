# Migrating from v2 to v3

v3 is a breaking release. The workflow shape survives (prepare, rank, vote,
plot) but every return type, several defaults, and the L1 classification
scores themselves changed.

## Why rankings differ, not just the API

- v2's L1 classification ranker recorded the largest C at which a
  coefficient was nonzero. On an L1 path that is the top of the grid for
  nearly every feature, so the scores were mostly ties and the resulting
  order was sort noise. v3 records the entry point (smallest C with a
  nonzero coefficient), which is the statistic the method was meant to
  compute. Expect genuinely different, and now meaningful, L1
  classification rankings.
- v2's L1 ranker crashed on 3+ classes and was nondeterministic (unseeded
  saga). v3 handles multiclass and is deterministic.
- Tree models are tuned by successive halving with balanced accuracy or
  RMSE instead of a small randomized search with plain accuracy, so their
  importances shift too.

## Breaking changes

| v2 | v3 |
|---|---|
| `feature_ranking(...) -> list[tuple[str, DataFrame]]` | returns [`RankingResult`](api.md#featurerankerrankingresult); per-method tables in `result.rankings` keyed by method name |
| `choices=["rf", ...]` parameter | renamed `methods=[...]` |
| ranking tables `[<method name>, "Score"]` | uniform `["feature", "score"]` |
| `voting(rankings, weights=[0.2, 0.4, 0.4])` | `voting(result, weights={"mi": 0.2, "f_test": 0.4, "l1": 0.4})`; missing keys default to 1.0 |
| `feature_ranking(..., save=True)` pickled a list | `result.save(path)` and `RankingResult.load(path)` |
| `voting(..., save=True)` wrote CSV | `voting(result).to_csv(path, index=False)` |
| `AssertionError` on bad input (stripped under `python -O`) | `ValueError` / `TypeError` always |
| `**kwargs` silently ignored typos | typed `options={...}` per method; unknown keys raise |
| `n_jobs` parallelized across methods (and could slow runs down) | `n_jobs` is a core budget spent inside each method |
| `plot_rankings(list_of_tuples)` drew synthetic positional bars | `plot_rankings(result)` draws real ranks as a dot plot |
| `hyper_param_search`, `make_ranking`, `l1_*_ranking`, `VALID_CHOICES` | removed; `methods=["l1"]` etc. cover the use cases |
| clustering module (5 functions) | removed; copy from the `2.0.0` tag if needed |
| Python >= 3.9 (nominal), scikit-learn >= 1.3 | Python >= 3.11, scikit-learn >= 1.8 |
| CC-BY-NC-SA-4.0 license | MIT |

## Before and after

Rank, vote, plot:

```python
# v2
rankings = feature_ranking(X, y, task="classification", choices=["mi", "f_test", "l1"])
scoring = voting(rankings, weights=[0.2, 0.4, 0.4])
plot_rankings(rankings)
plot_after_vote(scoring)

# v3
result = feature_ranking(X, y, task="classification", methods=["mi", "f_test", "l1"])
vote_table = voting(result, weights={"mi": 0.2, "f_test": 0.4, "l1": 0.4})
plot_rankings(result)
plot_after_vote(vote_table)
```

Reading one method's ranking:

```python
# v2
name, df = rankings[0]
top = df[name].head(10).tolist()

# v3
top = result.rankings["mi"]["feature"].head(10).tolist()
```

Persisting results:

```python
# v2
feature_ranking(X, y, save=True, save_path="rankings.pkl")

# v3
result = feature_ranking(X, y)
result.save("rankings.joblib")
```

## Dependency floors

| package | v2 floor | v3 floor |
|---|---|---|
| Python | 3.9 (declared; 3.10 in practice) | 3.11 |
| scikit-learn | 1.3.0 | 1.8 |
| xgboost | 1.7.0 | 2.0 |
| numpy | 1.24.0 | 1.26 |
| pandas | 2.0.0 | 2.1 |
| scipy | 1.10.0 | 1.11 |
| matplotlib | 3.7.0 | 3.8 |
| seaborn | 0.12.0 | 0.13 |
| joblib | 1.2.0 | 1.3 |
