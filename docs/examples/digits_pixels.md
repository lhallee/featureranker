# Handwritten digits from raw pixels

8x8 grayscale digits with each pixel as a named feature, so the ranking is a spatial saliency map.

Data: scikit-learn digits dataset, 1,797 images. 1,797 samples, 64 features. One five-method
ensemble ranking of the training split took 125 s and
provides every selector below; reductions fit on the same split. Three
standardized probes (accuracy: linear, kNN, RBF SVM) score the held-out
20%. Budgets anchor at the 99%-variance PCA count x = 41, stepping
x, x/2, x/4, 10, 1.

## Consensus ranking

| feature | score |
|---|---|
| pixel_4_1 | 3.292 |
| pixel_4_4 | 2.01 |
| pixel_2_5 | 1.855 |
| pixel_3_2 | 1.327 |
| pixel_7_4 | 1.183 |
| pixel_4_2 | 1.016 |
| pixel_5_2 | 0.8815 |
| pixel_3_4 | 0.737 |
| pixel_6_6 | 0.5702 |
| pixel_5_3 | 0.5382 |

![Aggregated importance](../images/digits_pixels_vote.png)

![Ranks by method](../images/digits_pixels_rankings.png)

## Ablation: selectors vs reductions (accuracy)

`select:<method>` keeps that method's top-k raw features
(`select:vote` is the ensemble); `dr:<method>` is a fitted reduction to
the same k.

| k | representation | linear | knn | svm |
|---|---|---|---|---|
| 64 | all features | 0.9722 | 0.9611 | 0.975 |
| 41 | dr:ICA | 0.9722 | 0.9694 | 0.9889 |
| 41 | dr:Isomap | 0.9806 | 0.975 | 0.9722 |
| 41 | dr:KernelPCA | 0.3417 | 0.3389 | 0.1722 |
| 41 | dr:PCA | 0.9722 | 0.9694 | 0.9889 |
| 41 | dr:RandProj | 0.9611 | 0.9667 | 0.9806 |
| 41 | dr:UMAP | 0.9694 | 0.9722 | 0.9556 |
| 41 | select:f_test | 0.9583 | 0.9778 | 0.9889 |
| 41 | select:l1 | 0.9583 | 0.9667 | 0.9889 |
| 41 | select:mi | 0.9583 | 0.975 | 0.9889 |
| 41 | select:rf | 0.9611 | 0.9833 | 0.9889 |
| 41 | select:vote | 0.9611 | 0.9778 | 0.9889 |
| 41 | select:xg | 0.9611 | 0.9722 | 0.9889 |
| 20 | dr:ICA | 0.95 | 0.9694 | 0.9889 |
| 20 | dr:Isomap | 0.9778 | 0.9778 | 0.9722 |
| 20 | dr:KernelPCA | 0.2722 | 0.2139 | 0.175 |
| 20 | dr:PCA | 0.95 | 0.9694 | 0.9889 |
| 20 | dr:RandProj | 0.9028 | 0.9222 | 0.9472 |
| 20 | dr:UMAP | 0.9778 | 0.9833 | 0.9639 |
| 20 | select:f_test | 0.9111 | 0.9417 | 0.9694 |
| 20 | select:l1 | 0.9 | 0.9444 | 0.9583 |
| 20 | select:mi | 0.9167 | 0.95 | 0.9722 |
| 20 | select:rf | 0.9417 | 0.9667 | 0.9778 |
| 20 | select:vote | 0.9417 | 0.9556 | 0.9722 |
| 20 | select:xg | 0.9222 | 0.9417 | 0.9583 |
| 10 | dr:ICA | 0.9167 | 0.95 | 0.9722 |
| 10 | dr:Isomap | 0.9722 | 0.9722 | 0.9778 |
| 10 | dr:KernelPCA | 0.2583 | 0.175 | 0.1806 |
| 10 | dr:PCA | 0.9167 | 0.95 | 0.9722 |
| 10 | dr:RandProj | 0.7556 | 0.825 | 0.8833 |
| 10 | dr:UMAP | 0.9722 | 0.9833 | 0.975 |
| 10 | dr:t-SNE | 0.9583 | 0.9833 | 0.9833 |
| 10 | select:f_test | 0.8444 | 0.9028 | 0.9306 |
| 10 | select:l1 | 0.8028 | 0.8389 | 0.8528 |
| 10 | select:mi | 0.8028 | 0.8444 | 0.8583 |
| 10 | select:rf | 0.8472 | 0.8972 | 0.9139 |
| 10 | select:vote | 0.8278 | 0.8833 | 0.8972 |
| 10 | select:xg | 0.8083 | 0.8222 | 0.8472 |
| 1 | dr:ICA | 0.3694 | 0.3556 | 0.375 |
| 1 | dr:Isomap | 0.5167 | 0.5444 | 0.5972 |
| 1 | dr:KernelPCA | 0.1444 | 0.1028 | 0.1278 |
| 1 | dr:PCA | 0.3694 | 0.3556 | 0.375 |
| 1 | dr:RandProj | 0.2583 | 0.2694 | 0.2806 |
| 1 | dr:UMAP | 0.5306 | 0.95 | 0.9139 |
| 1 | dr:t-SNE | 0.8611 | 0.9611 | 0.9 |
| 1 | select:f_test | 0.2306 | 0.2222 | 0.25 |
| 1 | select:l1 | 0.2306 | 0.2222 | 0.25 |
| 1 | select:mi | 0.2306 | 0.2222 | 0.25 |
| 1 | select:rf | 0.2417 | 0.2361 | 0.2722 |
| 1 | select:vote | 0.2306 | 0.2222 | 0.25 |
| 1 | select:xg | 0.2583 | 0.2417 | 0.2444 |

Notes: t-SNE has no out-of-sample transform, so it is fit on all rows
(label-blind) and split afterward, which favors it mildly; UMAP, kernel
PCA, Isomap, and ICA run at budgets up to 100 and t-SNE up to
10, beyond which their cost is impractical, while selection has
no budget ceiling. Cross-dataset findings: [the research report](report.md).

Regenerate with `python examples/selection_vs_reduction.py --name digits_pixels`.
