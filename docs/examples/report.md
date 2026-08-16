# Feature selection vs dimensionality reduction: 20 datasets

## Aim

When a model needs fewer input dimensions, the common reflex is a fitted
reduction (PCA and friends). featureranker offers the alternative of
keeping a ranked subset of the raw features. This report measures both
across 20 datasets spanning transformer embeddings, classical
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

| example | domain | task | n | p | x (99% PCA) |
|---|---|---|---|---|---|
| bioresponse | beyond NLP | classification | 3751 | 1776 | 819 |
| breast_cancer | beyond NLP | classification | 569 | 30 | 17 |
| diabetes | beyond NLP | regression | 442 | 10 | 8 |
| digits_pixels | beyond NLP | classification | 1797 | 64 | 54 |
| har_sensors | beyond NLP | classification | 4000 | 561 | 178 |
| madelon | beyond NLP | classification | 2600 | 500 | 470 |
| mnist_pixels | beyond NLP | classification | 4000 | 784 | 431 |
| chargram_langid | classical NLP | classification | 3600 | 2000 | 1395 |
| glove_sst2 | classical NLP | classification | 4000 | 300 | 219 |
| spambase_engineered | classical NLP | classification | 4601 | 57 | 54 |
| tfidf_newsgroups | classical NLP | classification | 2257 | 2000 | 1392 |
| tfidf_sst2 | classical NLP | classification | 4000 | 1774 | 1541 |
| modernbert_agnews | transformer NLP | classification | 4000 | 1536 | 1153 |
| modernbert_emotion | transformer NLP | classification | 4000 | 1536 | 1171 |
| modernbert_imdb | transformer NLP | classification | 3000 | 1536 | 997 |
| modernbert_offensive | transformer NLP | classification | 4000 | 1536 | 1133 |
| modernbert_sms_spam | transformer NLP | classification | 4000 | 1536 | 1153 |
| modernbert_sst2 | transformer NLP | classification | 4000 | 1536 | 1206 |
| modernbert_subjectivity | transformer NLP | classification | 4000 | 1536 | 1206 |
| modernbert_trec | transformer NLP | classification | 4000 | 1536 | 1198 |

## Results

### Ensemble selection vs each reduction

One duel per dataset, budget, and probe where both sides ran:

| reduction | duels | vote_wins | losses | mean_delta |
|---|---|---|---|---|
| ICA | 165 | 84 | 73 | -0.01 |
| Isomap | 165 | 102 | 62 | 0.003 |
| KernelPCA | 165 | 105 | 59 | 0.005 |
| PCA | 300 | 181 | 107 | 0.029 |
| RandProj | 297 | 259 | 32 | 0.066 |
| UMAP | 165 | 97 | 63 | -0.015 |
| t-SNE | 135 | 76 | 56 | -0.022 |

Mean score delta (vote selection minus reduction) by probe:

| reduction | knn | linear | svm |
|---|---|---|---|
| ICA | -0.004 | -0.015 | -0.011 |
| Isomap | 0.005 | -0.001 | 0.004 |
| KernelPCA | 0.008 | 0.004 | 0.003 |
| PCA | 0.08 | -0.005 | 0.013 |
| RandProj | 0.079 | 0.064 | 0.056 |
| UMAP | -0.041 | 0.015 | -0.018 |
| t-SNE | -0.057 | 0.02 | -0.029 |

### Ablating the ensemble against its own methods

Mean probe score and mean rank among the six selectors across every
dataset, budget, and probe (rank 1 = best of the six):

| selector | mean_score | mean_rank |
|---|---|---|
| vote | 0.719 | 3.218 |
| rf | 0.719 | 3.073 |
| xg | 0.708 | 3.817 |
| mi | 0.708 | 4.022 |
| f_test | 0.715 | 3.797 |
| l1 | 0.717 | 3.073 |

### Keeping at most 10 features

The vote's top rung at k <= 10 retains a median 92%
of the all-features linear-probe score (mean 89%,
worst 58% on mnist_pixels, best
117% on madelon).

### Cost

| method | median fit seconds |
|---|---|
| dr:ICA | 0.4 |
| dr:Isomap | 1.5 |
| dr:KernelPCA | 0.7 |
| dr:PCA | 0 |
| dr:RandProj | 0 |
| dr:UMAP | 12 |
| dr:t-SNE | 1.2 |
| ranking (all selectors) | 149.2 |

Ranking is a one-off cost that covers all six selectors at every budget;
each reduction pays per budget. Largest vote-selection advantage over PCA:
chargram_langid (k=1395, knn probe, 0.889
vs 0.203). Largest deficit: mnist_pixels
(k=10, svm probe, 0.516 vs
0.884).

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
