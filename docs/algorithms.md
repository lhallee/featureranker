# Algorithms

Each method produces one score per feature, higher meaning more important.
Scores from different methods live on different scales; only their ranks are
compared during voting.

## rf: random forest impurity importance

A `HalvingRandomSearchCV` tunes a random forest over a bounded space
(150 to 500 trees, depth caps, feature subsampling; see
[api.md](api.md#featurerankertreesearchoptions)) using successive halving:
candidate configurations compete on small row subsets and only winners
advance to larger ones. The best parameters then get exactly one full-data
fit, and the score is that model's `feature_importances_` (mean decrease in
impurity). Scoring during the search is balanced accuracy for classification
and negative RMSE for regression. Datasets too small to form halving rungs
fall back to an 8-candidate `RandomizedSearchCV`.

Impurity importances are computed on the training data and can inflate
high-cardinality features; the ensemble vote with four other methods is the
mitigation this package takes.

## xg: XGBoost gain importance

The same halving search and single final fit, over an XGBoost space with
log-uniform learning rate and regularization terms, `tree_method="hist"`.
Scores are xgboost's default `feature_importances_` (total gain share).
`options={"xg": {"estimator_kwargs": {"device": "cuda"}}}` moves training to
a GPU.

## mi: mutual information

`mutual_info_classif` / `mutual_info_regression`, the k-nearest-neighbor
estimator (default `n_neighbors=3`). Each feature scores its estimated
mutual information with the target in nats; independent features score
near 0. Above `max_samples` rows (default 100,000) the estimate runs
on a seeded row subsample, logged at INFO level, because the kNN cost grows
as n log n per feature while the estimate stabilizes well below the cap.
`options={"mi": {"max_samples": None}}` forces the exact computation.

## f_test: ANOVA F statistic

`f_classif` (between-class over within-class variance) for classification,
`f_regression` (squared correlation converted to an F statistic) for
regression. Constant features receive score 0 instead of NaN. This is the
fastest method and sees only linear (regression) or mean-shift
(classification) signal.

## l1: regularization-path entry points

The L1 score answers: how strong can the penalty be before the model drops
this feature? Features that survive stronger regularization matter more.

### Regression: entry alpha on the lasso path

For standardized features, the lasso path starts at
`alpha_max = max_j |x_j^T y| / n`, where every coefficient is zero, and
features enter one by one as alpha decreases. The score is each feature's
entry alpha, the largest penalty at which its coefficient is nonzero.
Features that never enter score 0.

Two routes compute the same quantity (`LassoOptions.strategy`):

- `"exact"`: the LARS algorithm on a precomputed standardized Gram matrix
  gives the exact path breakpoints, with no grid. Chosen automatically when
  `p <= max_gram_features` (default 4096) and `n * p^2 <= max_gram_flops`
  (default 4e12).
- `"grid"`: a dense warm-started coordinate-descent path
  (`n_alphas=200`, `eps=1e-4`, solver `tol=1e-6`) brackets each entry between
  consecutive grid points; the score is the geometric mean of the bracket.
  Used for wide data where the Gram matrix does not fit.

`"auto"` (default) picks between them; a numerically degenerate exact path
falls back to the grid with a logged warning.

### Classification: entry C from parallel waves

For L1 logistic regression the inverse penalty C plays the opposite role:
below `l1_min_c` every coefficient is zero, and features enter as C grows.
The score is `1 / entry_C`, where entry_C is the smallest C at which any of
the feature's class coefficients is nonzero (so earlier entry means a higher
score). Features absent even at the top of the search range score 0.

There is no exact path algorithm for logistic loss, so entry points are
bracketed by independent single-threaded fits fanned across cores in waves:

1. A coarse wave of `coarse_size` (default 16) log-spaced C values over
   `decades` (default 3) decades above `l1_min_c`, one parallel fit each.
   Every fit reveals the full active set, so one fit tightens the bracket of
   every feature at once.
2. One extension wave up to `max_extra_decades` further if some features
   have not appeared.
3. Refinement waves at the geometric midpoints of still-open brackets
   (widest first, deduplicated) until every bracket is narrower than
   `1 + rtol` (default 15% width) or the wave/fit budget runs out.

The solver is liblinear for binary problems up to `liblinear_max_n` rows
(default 50,000) and saga otherwise, including all multiclass problems.
Entry values from the two solvers differ slightly (liblinear approximates an
unpenalized intercept via `intercept_scaling=100`); force one with
`options={"l1": {"solver": "saga"}}` when comparing across dataset sizes.

## Voting

`voting` converts each method's scores into average ranks (exact ties share
the average of the positions they span), maps ranks to points, multiplies by
the method's weight, and sums per feature:

| method | points for rank r among n features |
|---|---|
| `reciprocal_rank` (default) | `w * 1 / r` |
| `borda` | `w * (n - r)` |
| `exponential` | `w * exp(-(r - 1) / max(n - 1, 1))` |

Reciprocal rank concentrates weight at the top of each ranking; Borda counts
every position linearly; exponential decays smoothly from 1 to `exp(-1)`
across the ranking (v2's variant spanned that same factor but never reached
1.0 and collapsed at p=1; v3 normalizes the exponent).

Ties everywhere are deterministic: tied scores within a method contribute
identical points, and displayed tables break remaining ties by feature name
with a stable sort.
