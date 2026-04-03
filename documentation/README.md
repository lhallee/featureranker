# API Documentation

## rankers.py

### `make_ranking(name: str, cols: list[str], importance: np.ndarray) -> pd.DataFrame`

Create a DataFrame ranking features by importance scores. Returns a DataFrame with columns `[name, "Score"]` sorted by score descending.

### `l1_regression_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame`

Rank features via L1-regularized linear regression (lasso path). Features are ranked by the maximum alpha at which their coefficients remain non-zero. Returns DataFrame with columns `["L1", "Score"]`.

### `l1_classification_ranking(X: pd.DataFrame, y: pd.Series, n_regularization_steps: int = 50) -> pd.DataFrame`

Rank features via L1-regularized logistic regression. Features are ranked by the maximum C value at which their coefficients remain non-zero. Returns DataFrame with columns `["L1", "Score"]`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_regularization_steps` | `int` | `50` | Number of C values to evaluate along the regularization path |

### `feature_ranking(X, y, task, choices, n_jobs, save, save_path, **kwargs) -> list[tuple[str, pd.DataFrame]]`

Run an ensemble of feature ranking methods in parallel.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | required | Feature matrix |
| `y` | `pd.Series` | required | Target vector |
| `task` | `str` | `"classification"` | `"classification"` or `"regression"` |
| `choices` | `list[str] \| None` | `None` | Methods to use. Default: all five (`["rf", "xg", "mi", "f_test", "l1"]`) |
| `n_jobs` | `int` | `1` | Parallel jobs for dispatching rankers. Use `-1` for all cores |
| `save` | `bool` | `False` | Save rankings to pickle file |
| `save_path` | `str \| None` | `None` | Path for pickle file (auto-generated if None) |
| `**kwargs` | | | Passed to rankers: `cv`, `n_iter`, `verbose`, `n_regularization_steps`, `search_n_jobs` |

### `voting(rankings, weights, method, save, save_path) -> pd.DataFrame`

Aggregate feature rankings using a weighted voting scheme.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rankings` | `list[tuple[str, pd.DataFrame]]` | required | Output from `feature_ranking()` |
| `weights` | `list[float] \| None` | `None` | Weight per method (default: equal weights) |
| `method` | `str` | `"reciprocal_rank"` | Voting method: `"reciprocal_rank"`, `"borda"`, or `"exponential"` |
| `save` | `bool` | `False` | Save to CSV |
| `save_path` | `str \| None` | `None` | Path for CSV file |

**Voting methods:**
- `"reciprocal_rank"`: `weight * (1 / rank)` -- higher ranks contribute more
- `"borda"`: `weight * (n_features - rank)` -- linear scoring
- `"exponential"`: `weight * exp(-rank / n_features)` -- exponential decay

---

## utils.py

### `sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame`

Replace non-word characters in column names with underscores. Returns a copy; does not mutate the original.

### `view_data(df: pd.DataFrame) -> pd.Series`

Return a Series of NaN percentages per column (only columns with NaNs). Also logs the information.

### `get_data(df, target, thresh, columns_to_drop, n_rows) -> tuple[pd.DataFrame, pd.Series]`

Prepare dataset by cleaning and encoding features.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | Input DataFrame |
| `target` | `str` | required | Name of the target column |
| `thresh` | `float` | `0.8` | Minimum fraction of non-NaN values to keep a column |
| `columns_to_drop` | `list[str] \| None` | `None` | Columns to drop before processing |
| `n_rows` | `int \| None` | `None` | Shuffle and sample this many rows |

Pipeline: drop columns -> drop high-NaN columns -> drop NaN rows -> sample -> remove constant columns -> label-encode categoricals.

### `hyper_param_search(X, y, model_name, task, cv, n_iter, n_jobs, verbose) -> dict`

Hyperparameter search using `RandomizedSearchCV`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | required | Feature matrix |
| `y` | `pd.Series` | required | Target vector |
| `model_name` | `str` | required | `"RandomForest"` or `"XGBoost"` |
| `task` | `str` | required | `"classification"` or `"regression"` |
| `cv` | `int` | `3` | Cross-validation folds |
| `n_iter` | `int` | `5` | Number of parameter settings sampled |
| `n_jobs` | `int` | `-1` | Parallel jobs for the search |
| `verbose` | `int` | `0` | Verbosity level |

---

## plots.py

All plot functions share a consistent interface with these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save` | `bool` | `False` | Save to PNG at 300 DPI |
| `save_path` | `str \| None` | `None` | Custom save path (auto-generated if None) |
| `show` | `bool` | `True` | Call `plt.show()` |
| `ax` | `plt.Axes \| None` | `None` | Existing axes to draw on (creates new figure if None) |

All return `plt.Axes`.

### `plot_correlations(predictions, labels, model_name, ...)`

Scatter plot of predictions vs true values with line of best fit, Pearson/Spearman correlations, and R2 score.

### `plot_confusion_matrix(c_matrix, labels, title, ...)`

Heatmap of a confusion matrix using seaborn.

### `plot_after_vote(scoring, title, ..., height_per_feature, highlight_feature)`

Horizontal bar chart of aggregated feature scores. `height_per_feature` controls plot height scaling. `highlight_feature` highlights a specific feature bar.

### `plot_rankings(rankings, title, ..., height_per_feature)`

Overlapping horizontal bar charts for multiple ranking methods with a legend.

---

## clustering.py

### `random_cluster_generator(n_samples, n_features, n_centers, std) -> np.ndarray`

Generate random clustered data using `sklearn.datasets.make_blobs`.

### `get_inertia(X: np.ndarray, k: int) -> float`

Compute within-cluster sum of squares for k clusters.

### `optimal_k_w_elbow(X: np.ndarray, max_k: int = 10) -> int`

Find optimal k using the elbow method (maximum distance from baseline). Requires `max_k >= 2`.

### `get_kmean_metrics(X: np.ndarray, k: int) -> tuple[float, float]`

Return `(inertia, silhouette_score)` for k clusters. Silhouette is 0.0 for k=1.

### `optimal_k_w_both(X: np.ndarray, max_k: int = 10) -> int`

Find optimal k using combined elbow + silhouette scoring. Requires `max_k >= 2`.
