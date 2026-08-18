# Quickstart

## Install

```bash
pip install featureranker
```

Requires Python >= 3.11. The heavy dependencies are scikit-learn >= 1.8,
xgboost >= 2.0, and datasets >= 2.19 (the Hugging Face loaders).

## Classification in five lines

```python
from sklearn.datasets import load_breast_cancer
from featureranker import feature_ranking, voting, plot_rankings, plot_after_vote

data = load_breast_cancer(as_frame=True)
result = feature_ranking(data.data, data.target, task="classification")
vote_table = voting(result)
plot_rankings(result, top_n=15)
plot_after_vote(vote_table, top_n=15)
```

`feature_ranking` returns a [`RankingResult`](api.md#featurerankerrankingresult).
The pieces you will use most:

```python
result.methods            # ("rf", "xg", "mi", "f_test", "l1")
result.rankings["rf"]     # DataFrame ["feature", "score"], best first
result.rank_matrix()      # features x methods, average ranks, 1 = best
result.diagnostics["l1"]  # solver, fit counts, entry thresholds
```

## Preparing a raw DataFrame

`get_data` cleans a raw frame: it drops columns you name, drops columns with
too many missing values, drops rows with remaining missing values, removes
constant columns, and encodes categorical, boolean, and datetime columns plus
a non-numeric target.

```python
import pandas as pd
from featureranker import get_data, view_data

df = pd.read_csv("my_data.csv")
view_data(df)                        # missing-value report per column
X, y = get_data(df, target="label")  # cleaned features and encoded target
```

`columns_to_drop` takes exact names or glob patterns:
`columns_to_drop=["id", "target_*"]` drops the `id` column and every
`target_`-prefixed column (a pattern never drops the target itself).

Categorical columns one-hot expand into named sub-features by default: a
`color` column with values `blue` and `red` becomes `color-blue` and
`color-red`, injected where `color` was, so the rankings speak in category
memberships. `encoding="label"` keeps single label-encoded columns instead,
and columns with more than `max_categories` unique values (default 64) fall
back to label encoding automatically.

## Loading a Hugging Face dataset

`get_hf_data` takes a Hub path, downloads it, converts it to pandas, and
runs the same cleaning: name the label column, list what to exclude, and
every remaining column becomes a feature.

```python
from featureranker import feature_ranking, get_hf_data, voting

X, y = get_hf_data(
    "scikit-learn/adult-census-income",
    target="income",
    columns_to_drop=["fnlwgt"],
)
result = feature_ranking(X, y, task="classification")
voting(result).head(10)
```

When the dataset also carries validation or test splits, they are found
automatically and the return becomes a `DataSplits`, cleaned and encoded
jointly so the feature columns match across splits. Rank and fit on train,
tune on valid, report on test:

```python
splits = get_hf_data("org/dataset-with-splits", target="labels")
result = feature_ranking(splits.X_train, splits.y_train)
fit = result.fit_convex(
    splits.X_train, splits.y_train, top_n=15,
    valid=splits.valid, test=splits.test,
)
fit.metrics          # {"train": ..., "valid": ..., "test": ...}
```

`load_hf_dataset(path)` returns the raw DataFrame when you want to inspect
or transform it before `get_data`, and `hf_login()` authenticates for
private or gated datasets.

## One score from the best features

After ranking, `fit_convex` turns the strongest features into a single
scoring function: weights >= 0 that sum to one, fit by least squares, so
each weight is that feature's share of the score. Threshold with `top_n`
when the consensus shows a clear cut:

```python
result = feature_ranking(X, y, task="classification")
fit = result.fit_convex(X, y, top_n=10)   # combine the top 10 consensus features

fit.table()          # ["feature", "weight"], largest first
fit.metrics          # AUC or R2 per split: train, plus valid/test when given
fit.method_metrics   # the same metrics refit on each method's own top 10
scores = fit.predict(X_new)  # rank new rows with the fitted combination
```

Pass `valid=(X_valid, y_valid)` and `test=(X_test, y_test)` to score
held-out splits the fit never trains on.

The fit is deterministic and globally optimal.
Classification must be binary; the score ranks rows by class membership.
Features are z-scored internally by default so mixed scales cannot dominate
the fit; pass `standardize=False` when your features already share a scale
and the score should be the weighted average of the raw values. A
maximum-entropy term (`entropy`, default 0.1) keeps every weight strictly
positive and the optimum unique even for duplicated features;
`entropy=0` recovers plain least squares, where redundant features get
exact zero weights.

## Regression

```python
from sklearn.datasets import load_diabetes
from featureranker import feature_ranking, voting

data = load_diabetes(as_frame=True)
result = feature_ranking(data.data, data.target, task="regression")
vote_table = voting(result, method="borda")
```

## Unnamed features: embeddings and hidden states

`X` does not need column names. Pass a 2D numpy array (a transformer
embedding matrix, pooled hidden states, any dense representation) and
features get stable generated IDs whose integer suffix is the column index:

```python
import numpy as np
from featureranker import feature_ranking

E = np.load("pooled_embeddings.npy")   # (n_samples, n_dims)
labels = np.load("labels.npy")         # (n_samples,)
result = feature_ranking(E, labels, task="classification")
result.rankings["l1"].head(5)          # features named f0000, f0001, ...
```

With thousands of dimensions, only a handful usually matter; the plots
default to the top 30-40 features, so the important dimensions surface
without any manual selection. See the
[ModernBERT sentiment example](examples/modernbert_sentiment.md) for a full
run on 1,536 pooled transformer dimensions.

## Choosing methods and weights

```python
from featureranker import feature_ranking, voting

result = feature_ranking(X, y, methods=["mi", "f_test", "l1"])
vote_table = voting(result, weights={"l1": 2.0, "mi": 0.5})
```

Weights are keyed by method name; missing keys default to 1.0 and unknown
keys raise `ValueError`.

## Probe metrics and auto-tuned vote weights

`feature_ranking` automatically evaluates every method's ranking with a
shared cross-validated linear probe over top-k cuts (disable with
`probe=False`). `result.probe_table()` reports the score per method and
cut plus a normalized skill, and `weights="auto"` turns those skills into
vote weights, so more predictive methods vote harder:

```python
result = feature_ranking(X, y)
result.probe_table()                     # methods x top-k scores, skill
vote_table = voting(result, weights="auto")
```

## Tuning a method

Each method takes a typed options object, passed as a dict or as the
dataclass:

```python
from featureranker import feature_ranking, LogisticL1Options

result = feature_ranking(
    X, y,
    methods=["l1"],
    options={"l1": LogisticL1Options(rtol=0.05, max_fits=256)},
)
```

The option sets are listed per method in [api.md](api.md); what they control
is explained in [algorithms.md](algorithms.md) and
[performance.md](performance.md).

## Saving and loading

```python
from featureranker import RankingResult

result.save("cancer_rankings.joblib")
restored = RankingResult.load("cancer_rankings.joblib")
assert restored.equals(result)
```

Export any table with pandas: `voting(result).to_csv("ranking.csv", index=False)`.
