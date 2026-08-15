"""Breast cancer classification example; regenerates docs/examples/breast_cancer.md."""

import matplotlib

matplotlib.use("Agg")

from sklearn.datasets import load_breast_cancer

from _pages import image_path, md_table, save_page
from featureranker import (
    feature_ranking,
    plot_after_vote,
    plot_rank_heatmap,
    plot_rankings,
    voting,
)


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    result = feature_ranking(data.data, data.target, task="classification")
    vote_table = voting(result)

    plot_rankings(
        result, top_n=15, show=False, save=True,
        save_path=image_path("breast_cancer_rankings.png"),
    )
    plot_rank_heatmap(
        result, top_n=15, show=False, save=True,
        save_path=image_path("breast_cancer_heatmap.png"),
    )
    plot_after_vote(
        vote_table, top_n=15, show=False, save=True,
        highlight_feature=vote_table["feature"].iloc[0],
        save_path=image_path("breast_cancer_vote.png"),
    )

    seconds = {m: result.diagnostics[m]["seconds"] for m in result.methods}
    page = f"""# Breast cancer classification

The scikit-learn breast cancer dataset: {result.n_samples} tumors,
{result.n_features} named morphology features, binary malignancy label. The
full five-method ensemble ran in {sum(seconds.values()):.0f} s
(rf {seconds['rf']:.0f} s, xg {seconds['xg']:.0f} s, l1 {seconds['l1']:.1f} s,
mi {seconds['mi']:.1f} s, f_test {seconds['f_test']:.2f} s).

```python
from sklearn.datasets import load_breast_cancer
from featureranker import feature_ranking, voting

data = load_breast_cancer(as_frame=True)
result = feature_ranking(data.data, data.target, task="classification")
vote_table = voting(result)
```

## Consensus

{md_table(vote_table, 10)}

![Aggregated feature importance](../images/breast_cancer_vote.png)

## Where the methods agree and disagree

The dot plot shows each method's rank per feature; tight rows are consensus,
spread rows are disagreement (the L1 path disagrees with the tree models
about `mean area`, which is nearly collinear with `worst area`):

![Feature ranks by method](../images/breast_cancer_rankings.png)

![Rank heatmap](../images/breast_cancer_heatmap.png)

Regenerate this page and its images with `python examples/breast_cancer.py`.
"""
    print("wrote", save_page("breast_cancer", page))


if __name__ == "__main__":
    main()
