# Quickstart

## Install

```bash
pip install featureranker
```

Requires Python >= 3.11. The heavy dependencies are scikit-learn >= 1.8 and
xgboost >= 2.0.

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

## Regression

```python
from sklearn.datasets import load_diabetes
from featureranker import feature_ranking, voting

data = load_diabetes(as_frame=True)
result = feature_ranking(data.data, data.target, task="regression")
vote_table = voting(result, method="borda")
```

## Choosing methods and weights

```python
from featureranker import feature_ranking, voting

result = feature_ranking(X, y, methods=["mi", "f_test", "l1"])
vote_table = voting(result, weights={"l1": 2.0, "mi": 0.5})
```

Weights are keyed by method name; missing keys default to 1.0 and unknown
keys raise `ValueError`.

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
