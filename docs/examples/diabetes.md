# Diabetes regression

The scikit-learn diabetes dataset: 442 patients, 10
standardized clinical features, a continuous disease-progression target.
The L1 method used its exact route (13 exact
path breakpoints), so entry alphas carry no grid quantization.

```python
from sklearn.datasets import load_diabetes
from featureranker import feature_ranking, voting

data = load_diabetes(as_frame=True)
result = feature_ranking(data.data, data.target, task="regression")
vote_table = voting(result)
```

## Consensus

| feature | score |
|---|---|
| bmi | 4 |
| s5 | 3.5 |
| bp | 1.393 |
| s4 | 1.208 |
| s6 | 1 |
| s3 | 0.9595 |
| sex | 0.725 |
| s1 | 0.7107 |
| s2 | 0.5873 |
| age | 0.5611 |

![Aggregated feature importance](../images/diabetes_vote.png)

Body mass index and one serum triglyceride measure dominate every method;
the dot plot shows the rest is where methods differ:

![Feature ranks by method](../images/diabetes_rankings.png)

Regenerate this page and its images with `python examples/diabetes.py`.
