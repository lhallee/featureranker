"""Generate the selection-vs-reduction research report from results.csv.

    python examples/write_report.py
"""

import numpy as np
import pandas as pd

from _pages import PAGES, md_table, save_page

SELECTORS = ["select:vote", "select:rf", "select:xg", "select:mi", "select:f_test", "select:l1"]


def _duels(rows: pd.DataFrame) -> pd.DataFrame:
    """Ensemble-vote selection vs each reduction at shared budgets, per probe."""
    vote = rows[rows.representation == "select:vote"]
    records = []
    for reduction in sorted(rows[rows.family == "dr"].representation.unique()):
        merged = vote.merge(
            rows[rows.representation == reduction],
            on=["example", "k", "probe"],
            suffixes=("_sel", "_red"),
        )
        for probe, block in merged.groupby("probe"):
            delta = block.score_sel - block.score_red  # (n_duels,)
            records.append({
                "reduction": reduction.removeprefix("dr:"),
                "probe": probe,
                "duels": len(block),
                "vote wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "mean delta": round(float(delta.mean()), 3),
            })
    return pd.DataFrame(records)


def _selector_ablation(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean score and mean within-group rank of each selector."""
    selectors = rows[rows.family == "select"].copy()
    selectors["rank"] = selectors.groupby(["example", "k", "probe"])["score"].rank(
        ascending=False, method="average"
    )
    summary = (
        selectors.groupby("representation")
        .agg(mean_score=("score", "mean"), mean_rank=("rank", "mean"))
        .round(3)
        .reindex(SELECTORS)
        .reset_index()
        .rename(columns={"representation": "selector"})
    )
    summary["selector"] = summary["selector"].str.removeprefix("select:")
    return summary


def _retention(rows: pd.DataFrame) -> pd.Series:
    """Per example: linear-probe vote score at the rung nearest 10 over all features."""
    linear = rows[rows.probe == "linear"]
    values = {}
    for example, block in linear.groupby("example"):
        baseline = block.loc[block.family == "all", "score"].iloc[0]
        small = block[(block.representation == "select:vote") & (block.k <= 10)]
        if len(small) and baseline > 0:
            values[example] = small.loc[small.k.idxmax()].score / baseline
    return pd.Series(values)


def main() -> None:
    rows = pd.read_csv(PAGES / "results.csv")

    datasets = (
        rows.groupby("example")
        .first()
        .reset_index()[["example", "domain", "task", "n", "p", "pca_99"]]
        .sort_values(["domain", "example"])
        .rename(columns={"pca_99": "x (99% PCA)"})
    )
    n_examples = len(datasets)

    duels = _duels(rows)
    duel_pivot = duels.pivot_table(
        index="reduction", columns="probe",
        values=["vote wins", "losses", "mean delta"],
    )
    duel_summary = (
        duels.groupby("reduction")
        .agg(duels=("duels", "sum"), vote_wins=("vote wins", "sum"),
             losses=("losses", "sum"), mean_delta=("mean delta", "mean"))
        .round(3)
        .reset_index()
    )
    per_probe_delta = (
        duels.pivot_table(index="reduction", columns="probe", values="mean delta")
        .round(3)
        .reset_index()
    )

    ablation = _selector_ablation(rows)
    retention = _retention(rows)

    timing = (
        rows[rows.family != "all"]
        .assign(method=lambda r: r.representation.str.replace("select:.*", "ranking (all selectors)", regex=True))
        .groupby("method")["fit_seconds"]
        .median()
        .round(1)
        .rename("median fit seconds")
        .reset_index()
    )

    pca = rows[rows.representation == "select:vote"].merge(
        rows[rows.representation == "dr:PCA"],
        on=["example", "k", "probe"], suffixes=("_sel", "_red"),
    )
    pca["delta"] = pca.score_sel - pca.score_red
    best, worst = pca.loc[pca.delta.idxmax()], pca.loc[pca.delta.idxmin()]

    report = f"""# Feature selection vs dimensionality reduction: {n_examples} datasets

## Aim

When a model needs fewer input dimensions, the common reflex is a fitted
reduction (PCA and friends). featureranker offers the alternative of
keeping a ranked subset of the raw features. This report measures both
across {n_examples} datasets spanning transformer embeddings, classical
NLP feature spaces, and non-text data, and ablates the ensemble against
its five individual ranking methods.

## Protocol

Each dataset gets a stratified 80/20 split (random_state=42 everywhere).
One `feature_ranking` ensemble pass on the training split yields six
selectors (each method's own top-k plus the weighted vote). Seven
reductions (PCA, FastICA, Gaussian random projection, RBF kernel PCA,
Isomap, UMAP, t-SNE) fit on the same split at the same budgets. Three
standardized probes score every representation on the held-out 20%:
logistic regression (ridge for the one regression dataset; metric R2),
10-nearest-neighbors, and an RBF SVM, capturing linear, local, and kernel
separability. Budgets anchor at x, the 99%-variance PCA component count,
stepping x, x/2, x/4, 10, 1.

Two caveats stated once: t-SNE has no out-of-sample transform, so it is
fit on all rows label-blind and split afterward, which favors it mildly;
and UMAP, kernel PCA, Isomap, and ICA run only at budgets up to 100 and
t-SNE up to 10 because their cost above that is impractical, while
selection has no ceiling (any k is a column slice once ranked). All
numbers live in `docs/examples/results.csv`.

## Datasets

{md_table(datasets, len(datasets))}

## Results

### Ensemble selection vs each reduction

One duel per dataset, budget, and probe where both sides ran:

{md_table(duel_summary, len(duel_summary))}

Mean score delta (vote selection minus reduction) by probe:

{md_table(per_probe_delta, len(per_probe_delta))}

### Ablating the ensemble against its own methods

Mean probe score and mean rank among the six selectors across every
dataset, budget, and probe (rank 1 = best of the six):

{md_table(ablation, len(ablation))}

### Keeping at most 10 features

The vote's top rung at k <= 10 retains a median {retention.median():.0%}
of the all-features linear-probe score (mean {retention.mean():.0%},
worst {retention.min():.0%} on {retention.idxmin()}, best
{retention.max():.0%} on {retention.idxmax()}).

### Cost

{md_table(timing, len(timing))}

Ranking is a one-off cost that covers all six selectors at every budget;
each reduction pays per budget. Largest vote-selection advantage over PCA:
{best.example} (k={int(best.k)}, {best.probe} probe, {best.score_sel:.3f}
vs {best.score_red:.3f}). Largest deficit: {worst.example}
(k={int(worst.k)}, {worst.probe} probe, {worst.score_sel:.3f} vs
{worst.score_red:.3f}).

## Reading the per-example pages

Every dataset has a page in this folder with its consensus ranking, plots,
and the full budget-by-budget ablation table, regenerated by
`python examples/selection_vs_reduction.py --name <example>`.

## Limitations

Single split per dataset, so no confidence intervals; probes and
reductions run with library defaults, as does the ranking ensemble;
datasets capped near 4,000 samples; the t-SNE and budget-ceiling caveats
above.

## Conclusion

Ranked raw features are a strong default when shrinking input width:
selection keeps original, interpretable dimensions, needs no fitted
transform at inference, scales to any budget after one ranking pass, and
the tables above quantify how often that simplicity also wins the probe.
"""
    save_page("report", report)
    print("wrote report over", n_examples, "examples and", len(rows), "probe scores")


if __name__ == "__main__":
    main()
