# API reference

Everything importable from `featureranker`. Headings are fully qualified
names; `grep -n "### featureranker\." docs/api.md` lists the surface. All
user-input validation raises `ValueError` or `TypeError` with the allowed
values in the message.

## Ranking

### featureranker.feature_ranking

```python
def feature_ranking(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    methods: Sequence[str] | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    dtype: Literal["float32", "float64"] = "float32",
    options: Mapping[str, Mapping[str, object] | object] | None = None,
    probe: bool = True,
) -> RankingResult
```

Runs the chosen methods sequentially over one shared numpy conversion of X
and returns a [`RankingResult`](#featurerankerrankingresult). With
`probe=True` each finished ranking is then evaluated by a shared
cross-validated linear probe (standardized logistic regression or ridge,
3-fold, over top-k cuts 1, 2, 4, ... 64, on at most 10,000 seeded rows):
the per-method reports land in `diagnostics[method]["probe"]` with the
score at each cut, their mean, and a normalized skill in [0, 1], readable
via [`probe_table`](#featurerankerrankingresult) and consumed by
[`voting(weights="auto")`](#featurerankervoting).

| parameter | meaning |
|---|---|
| `X` | numeric feature matrix: a DataFrame keeps its column names; a bare 2D numpy array (embeddings, hidden states) gets generated IDs from [`generated_feature_names`](#featurerankergenerated_feature_names); encode raw frames first with [`get_data`](#featurerankerget_data) |
| `y` | target; classification targets of any dtype are label-encoded, the original labels land in `result.classes` |
| `methods` | subset of [`METHODS`](#featurerankermethods) in run order; default all five |
| `n_jobs` | total core budget, -1 = all cores (see [performance.md](performance.md)) |
| `random_state` | seed threaded into every stochastic component |
| `dtype` | dtype of the shared feature array; float32 halves memory |
| `options` | method key to option dict or options dataclass, e.g. `{"mi": {"max_samples": None}}` |

Raises `ValueError` for: unknown task, dtype, method, or option key; empty or
duplicated methods; options for methods that will not run; duplicate column
names; NaN or infinite values in X; length mismatch between X and y; missing
values in y; fewer than 2 classes; constant regression target; bad `n_jobs`.
Raises `TypeError` when X is not a DataFrame or an options value has the
wrong type.

### featureranker.METHODS

```python
METHODS: tuple[str, ...] = ("rf", "xg", "mi", "f_test", "l1")
```

Method keys, in default run order. See [algorithms.md](algorithms.md).

### featureranker.generated_feature_names

```python
def generated_feature_names(n_features: int) -> tuple[str, ...]
```

The stable zero-padded IDs assigned to unnamed feature matrices
(`f0000`, `f0001`, ...). Exposed so downstream code can map a ranked ID back
to a column index: the integer suffix is the position in the input matrix.

### featureranker.RankingResult

```python
@dataclass(frozen=True, eq=False)
class RankingResult:
    task: str
    feature_names: tuple[str, ...]
    n_samples: int
    n_features: int
    rankings: dict[str, pd.DataFrame]
    diagnostics: dict[str, dict[str, object]]
    classes: tuple[object, ...] | None
    random_state: int
    version: str
```

Every table in `rankings` has columns `["feature", "score"]`, sorted best
first, higher score = more important, ties broken deterministically by
feature name. `classes` holds the original class labels for classification
(index = encoded value) and None for regression. `diagnostics` carries
method internals (solver choices, fit counts, best hyperparameters, raw L1
entry thresholds, per-method wall seconds) and is excluded from `equals`.

| member | meaning |
|---|---|
| `methods` | property; tuple of method keys in run order |
| `score_matrix() -> pd.DataFrame` | features x methods raw scores, rows in `feature_names` order |
| `rank_matrix() -> pd.DataFrame` | features x methods average ranks, 1 = best, ties share ranks |
| `probe_table() -> pd.DataFrame` | probe scores per method and top-k cut plus mean score and skill; raises `ValueError` without probe reports |
| `fit_convex(X, y, top_n=None, ...)` | fit a convex combination of the top consensus features, see [Convex combination](#featurerankerrankingresultfit_convex) |
| `equals(other) -> bool` | exact equality of rankings and metadata, diagnostics excluded |
| `save(path) -> None` | serialize with joblib |
| `RankingResult.load(path)` | classmethod; raises `ValueError` on unreadable files or wrong payload, warns on version mismatch |

### featureranker.voting

```python
def voting(
    result: RankingResult | Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | Literal["auto"] | None = None,
    method: Literal["reciprocal_rank", "borda", "exponential"] = "reciprocal_rank",
) -> pd.DataFrame
```

Aggregates rankings into one `["feature", "score"]` table, best first.
Accepts a `RankingResult` or any mapping of name to ranking table with the
standard columns. Weights are keyed by method name; missing keys default to
1.0, unknown keys raise `ValueError`, non-numeric weights raise `TypeError`.
`weights="auto"` weights each method by its probe skill, so more predictive
methods vote harder; it needs a `RankingResult` carrying the probe reports
from `feature_ranking(probe=True)` and raises `ValueError` otherwise. When
every method probes at chance level the weights fall back to equal.
Formulas in [algorithms.md](algorithms.md#voting).

## Convex combination

A convex combination scores each row as `X @ w` with weights `w >= 0` that
sum to one, so every weight reads as a feature's share of one interpretable
scoring function. The weights minimize the mean squared error of the score
against the target (for classification, against the 0/1 encoded labels, so
the score ranks rows by class membership rather than calibrated
probability). The problem is a convex quadratic program solved from a
uniform start, so the fit is deterministic and globally optimal.

`standardize=True` (the default) z-scores each feature before fitting and
stores the transform on the returned `ConvexFit`, which `predict` applies.
Standardized weights are unit-free shares, the right reading when features
carry different units or scales. Pass `standardize=False` when the
features are already commensurate (sub-scores of a ranking scheme, model
outputs on a shared scale) and the score should be the weighted average of
the raw values.

`entropy` (default 0.1) adds a maximum-entropy term
`entropy * Var(y) * sum(w * log w)` to the objective. The entropy gradient
diverges at the simplex boundary, so the optimum is strictly interior:
every weight stays strictly positive, and the fit is unique even when
features duplicate each other (a plain least-squares fit is ill-posed
there and an entropy fit splits the weight symmetrically). Scaling by
`Var(y)` makes the strength invariant to the units of the target.
`entropy=0` recovers plain least squares, where redundant features get
exact zero weights.

### featureranker.RankingResult.fit_convex

```python
def fit_convex(
    self,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    top_n: int | None = None,
    weights: Mapping[str, float] | Literal["auto"] | None = None,
    vote_method: Literal["reciprocal_rank", "borda", "exponential"] = "reciprocal_rank",
    standardize: bool = True,
    entropy: float = 0.1,
) -> ConvexFit
```

Runs [`voting`](#featurerankervoting) with the given `weights` and
`vote_method`, keeps the `top_n` consensus features (all when None), and
fits the combination on those columns of X. Every ranking method's own
top_n selection is also fitted and those metrics land in
`method_metrics` beside the returned `"ensemble"` fit, so the consensus
selection is directly comparable against each method's. X must carry the
same features that produced the result (same DataFrame columns, or a numpy
matrix of the same width), though the rows may differ, so the combination
can be fit on held-out data. A `top_n` above the feature count clamps to
all features. Raises `ValueError` when X does not match the ranked
features, `top_n` is below 1, `entropy` is negative, or the task is
classification with more than 2 classes.

### featureranker.fit_convex

```python
def fit_convex(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    standardize: bool = True,
    entropy: float = 0.1,
) -> ConvexFit
```

The standalone form: fits the combination over every column of X, no
ranking required. Input rules match
[`feature_ranking`](#featurerankerfeature_ranking); classification must be
binary.

### featureranker.ConvexFit

```python
@dataclass(frozen=True, eq=False)
class ConvexFit:
    task: str
    feature_names: tuple[str, ...]
    weights: np.ndarray
    feature_means: np.ndarray | None
    feature_stds: np.ndarray | None
    metric_name: str
    metric_value: float
    diagnostics: dict[str, object]
    method_metrics: dict[str, float] | None = None
```

`weights` aligns with `feature_names`, is nonnegative, and sums to one.
`feature_means`/`feature_stds` hold the stored standardization (None for a
raw fit). `metric_name`/`metric_value` hold R2 for regression or ROC AUC
for binary classification, computed on the fitting data. `diagnostics`
carries solver internals (convergence, iterations, mean squared error).
`method_metrics` is filled by `RankingResult.fit_convex`: the same metric
refit on each ranking method's own top_n selection plus the returned
`"ensemble"` fit; None for a standalone fit.

| member | meaning |
|---|---|
| `predict(X) -> np.ndarray` | score rows: a DataFrame is matched by column name, a numpy array must carry exactly the fitted features in order |
| `table() -> pd.DataFrame` | weights as `["feature", "weight"]`, largest first |

## Method options

Pass instances or plain dicts through `feature_ranking(options=...)`.
Unknown keys raise `ValueError` listing the valid set.

### featureranker.TreeSearchOptions

```python
@dataclass(frozen=True)
class TreeSearchOptions:
    cv: int = 3
    n_candidates: int = 27
    factor: int = 3
    scoring: str | None = None
    estimator_kwargs: Mapping[str, object] = {}
```

For `rf` and `xg`. `scoring=None` means balanced accuracy (classification)
or negative RMSE (regression). `estimator_kwargs` reach both the searched
and the final estimator; `{"device": "cuda"}` enables GPU XGBoost.

### featureranker.LassoOptions

```python
@dataclass(frozen=True)
class LassoOptions:
    strategy: Literal["auto", "exact", "grid"] = "auto"
    n_alphas: int = 200
    eps: float = 1e-4
    tol: float = 1e-6
    max_gram_features: int = 4096
    max_gram_flops: float = 4e12
```

For `l1` on regression tasks. `auto` picks `exact` when
`p <= max_gram_features` and `n * p^2 <= max_gram_flops`. `n_alphas`, `eps`,
and `tol` shape the grid route.

### featureranker.LogisticL1Options

```python
@dataclass(frozen=True)
class LogisticL1Options:
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
```

For `l1` on classification tasks. `auto` uses liblinear for binary problems
up to `liblinear_max_n` rows, saga otherwise; forcing liblinear on 3+
classes raises `ValueError`. `rtol` is the bracket width at which an entry C
counts as resolved.

### featureranker.MutualInfoOptions

```python
@dataclass(frozen=True)
class MutualInfoOptions:
    n_neighbors: int = 3
    max_samples: int | None = 100_000
```

For `mi`. Above `max_samples` rows the estimate runs on a seeded subsample;
None forces the exact computation.

## Data preparation

### featureranker.get_data

```python
def get_data(
    df: pd.DataFrame,
    target: str,
    thresh: float = 0.8,
    columns_to_drop: list[str] | None = None,
    n_rows: int | None = None,
    random_state: int = 42,
    encoding: Literal["onehot", "label"] = "onehot",
    max_categories: int | None = 64,
) -> tuple[pd.DataFrame, pd.Series]
```

Cleans a raw frame in a fixed order: drop `columns_to_drop`, drop feature
columns with less than `thresh` fraction of present values, drop rows with
remaining missing values, optionally shuffle-sample `n_rows`, drop constant
columns, then encode feature columns and a non-numeric target.
`columns_to_drop` entries are exact names or glob patterns: `"target_*"`
drops every matching column, a pattern never drops the target itself, and
a pattern matching nothing raises to catch typos.

Encoding rules: object/string/category feature columns one-hot expand into
one 0/1 sub-feature per unique value in the data, named
`"{column}-{value}"` (`color-blue` for strings, `color-0` for integer
categories) and injected at the parent column's position, so ranked
sub-features read directly as category memberships. Columns with more than
`max_categories` unique values fall back to a single label-encoded column
(`max_categories=None` removes the cap); `encoding="label"` restores the
previous behavior of label-encoding every categorical column. Booleans
become single 0/1 columns, datetime and timedelta columns become int64
nanoseconds, and a non-numeric target is label-encoded.

Raises `ValueError` for a missing target, unknown drop columns, a drop
pattern matching no columns, the target listed in `columns_to_drop`,
`thresh` outside (0, 1], unknown `encoding`, `max_categories` below 2,
`n_rows` beyond the cleaned length, or a sub-feature name colliding with
an existing column.

### featureranker.view_data

```python
def view_data(df: pd.DataFrame) -> pd.Series
```

Returns the percentage of missing values per column, restricted to columns
that have any, and logs the same report.

### featureranker.sanitize_column_names

```python
def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame
```

Returns a copy whose column names have non-word characters replaced with
underscores.

## Hugging Face

### featureranker.get_hf_data

```python
def get_hf_data(
    path: str,
    target: str,
    split: str = "train",
    name: str | None = None,
    thresh: float = 0.8,
    columns_to_drop: list[str] | None = None,
    n_rows: int | None = None,
    random_state: int = 42,
    encoding: Literal["onehot", "label"] = "onehot",
    max_categories: int | None = 64,
    **load_kwargs,
) -> tuple[pd.DataFrame, pd.Series]
```

Downloads one split of a Hub dataset and returns `(features, target)` ready
for [`feature_ranking`](#featurerankerfeature_ranking). `target` names the
label column, `columns_to_drop` excludes columns such as ids or free text
(exact names or glob patterns like `"target_*"`), and every remaining
column becomes a feature. The frame goes through
[`get_data`](#featurerankerget_data), so cleaning, sampling, and categorical
encoding behave exactly as documented there; `load_kwargs` pass through to
`datasets.load_dataset` (revision, data_files, token, ...).

### featureranker.load_hf_dataset

```python
def load_hf_dataset(
    path: str,
    split: str = "train",
    name: str | None = None,
    **load_kwargs,
) -> pd.DataFrame
```

Downloads one split of a Hub dataset and returns it as a raw pandas
DataFrame, without any cleaning or encoding. `name` selects a configuration
for multi-config datasets. Raises `ValueError` when `split` resolves to
multiple splits.

### featureranker.hf_login

```python
def hf_login(token: str | None = None) -> None
```

Authenticates with the Hugging Face Hub for private or gated datasets.
Without a token this opens the interactive Hub prompt; with a token it
stores the credential the same way `hf auth login` does. Public datasets
need no login.

## Plots

All plot functions share `save` (write a 300 dpi PNG), `save_path`
(default derived from the title), `show` (call `plt.show()`), and `ax`
(draw into an existing axes; saving targets that axes' own figure). They
return the `matplotlib.axes.Axes`.

### featureranker.plot_rankings

```python
def plot_rankings(
    result: RankingResult,
    top_n: int | None = 30,
    title: str = "Feature ranks by method",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.32,
    ax: plt.Axes | None = None,
) -> plt.Axes
```

Dot plot of per-method ranks: one row per feature ordered by average rank,
one fixed color per method, a hairline connector spanning each row's spread.
Row spread reads as method disagreement.

### featureranker.plot_rank_heatmap

```python
def plot_rank_heatmap(
    result: RankingResult,
    top_n: int | None = 40,
    title: str = "Rank heatmap",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.3,
    ax: plt.Axes | None = None,
) -> plt.Axes
```

Feature-by-method heatmap of ranks, darker = more important; cell
annotations switch off past 200 cells. The compact view for wide feature
sets.

### featureranker.plot_after_vote

```python
def plot_after_vote(
    vote_table: pd.DataFrame,
    top_n: int | None = 30,
    title: str = "Aggregated feature importance",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    height_per_feature: float = 0.3,
    highlight_feature: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes
```

Horizontal bars of `voting` scores. With `highlight_feature` set, that bar
keeps the accent color and the rest recede to gray.

### featureranker.plot_correlations

```python
def plot_correlations(
    predictions: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes
```

Prediction-vs-truth scatter with a fitted line and Pearson, Spearman, and R2
in the corner. For evaluating a downstream model, not the rankings.

### featureranker.plot_confusion_matrix

```python
def plot_confusion_matrix(
    c_matrix: np.ndarray,
    labels: np.ndarray,
    title: str = "Confusion matrix",
    save: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes
```

Annotated heatmap of a confusion matrix, darker = more samples.

## Removed in v3

These v2 names no longer exist; see
[migration_v2_to_v3.md](migration_v2_to_v3.md) for replacements:
`make_ranking`, `l1_regression_ranking`, `l1_classification_ranking`,
`hyper_param_search`, `VALID_CHOICES`, `random_cluster_generator`,
`get_inertia`, `optimal_k_w_elbow`, `get_kmean_metrics`, `optimal_k_w_both`.
