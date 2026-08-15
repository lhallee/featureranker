# featureranker documentation

featureranker ranks the features of any numeric feature matrix (a tabular
dataset, a transformer embedding, pooled hidden states) with five methods
(random forest, XGBoost, mutual information, ANOVA F-test, and L1
regularization paths), stores every ranking in one typed result object, and
aggregates them with weighted rank voting. These pages describe v3.0.0 on
Python >= 3.11.

## Where to look

| You want | Read |
|---|---|
| Install and first run | [quickstart.md](quickstart.md) |
| The math and score each method produces | [algorithms.md](algorithms.md) |
| Speed, n_jobs, adaptive strategies, determinism | [performance.md](performance.md) |
| Every public signature and exception | [api.md](api.md) |
| Upgrading code from v2 | [migration_v2_to_v3.md](migration_v2_to_v3.md) |
| Dev setup, tests, releases | [contributing.md](contributing.md) |

## Conventions in these docs

- API headings are fully qualified names, so
  `grep -n "### featureranker\." docs/api.md` enumerates the public surface.
- Every code example runs as pasted, imports included.
- All user-input errors are `ValueError` or `TypeError` with the allowed
  values in the message. There are no assertion-based checks.
- Ranking tables always have the columns `["feature", "score"]`, sorted best
  first, with scores oriented so higher means more important.
