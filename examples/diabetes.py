"""Diabetes regression example; regenerates docs/examples/diabetes.md."""

import matplotlib

matplotlib.use("Agg")

from sklearn.datasets import load_diabetes

from _pages import image_path, md_table, save_page
from featureranker import feature_ranking, plot_after_vote, plot_rankings, voting


def main() -> None:
    data = load_diabetes(as_frame=True)
    result = feature_ranking(data.data, data.target, task="regression")
    vote_table = voting(result)

    plot_rankings(
        result, show=False, save=True,
        save_path=image_path("diabetes_rankings.png"),
    )
    plot_after_vote(
        vote_table, show=False, save=True,
        save_path=image_path("diabetes_vote.png"),
    )

    l1 = result.diagnostics["l1"]
    page = f"""# Diabetes regression

The scikit-learn diabetes dataset: {result.n_samples} patients, 10
standardized clinical features, a continuous disease-progression target.
The L1 method used its {l1['strategy']} route ({l1['n_path_points']} exact
path breakpoints), so entry alphas carry no grid quantization.

```python
from sklearn.datasets import load_diabetes
from featureranker import feature_ranking, voting

data = load_diabetes(as_frame=True)
result = feature_ranking(data.data, data.target, task="regression")
vote_table = voting(result)
```

## Consensus

{md_table(vote_table, 10)}

![Aggregated feature importance](../images/diabetes_vote.png)

Body mass index and one serum triglyceride measure dominate every method;
the dot plot shows the rest is where methods differ:

![Feature ranks by method](../images/diabetes_rankings.png)

Regenerate this page and its images with `python examples/diabetes.py`.
"""
    print("wrote", save_page("diabetes", page))


if __name__ == "__main__":
    main()
