# featureranker

A lightweight Python package for robust feature importance ranking using an ensemble of methods with weighted voting.

The ensemble combines L1 penalization, random forests, XGBoost, ANOVA F-scores, and mutual information to rank feature importance for both classification and regression tasks.

Featured in:

- [Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life](https://www.nature.com/articles/s41598-023-28965-7) (Nature Scientific Reports, 2023)
- [cdsBERT - Extending Protein Language Models with Codon Awareness](https://www.biorxiv.org/content/10.1101/2023.09.15.558027v1.abstract) (bioRxiv, 2023)

## Installation

```bash
pip install featureranker
```

## Quick Start

```python
from sklearn.datasets import load_breast_cancer
from featureranker import get_data, feature_ranking, voting
from featureranker.plots import plot_after_vote, plot_rankings

# Load and prepare data
cancer = load_breast_cancer(as_frame=True)
df = cancer.data.merge(cancer.target, left_index=True, right_index=True)
X, y = get_data(df, target="target")

# Rank features using all five methods
rankings = feature_ranking(X, y, task="classification")

# Aggregate with weighted voting
scoring = voting(rankings)

# Visualize
plot_rankings(rankings, title="All methods")
plot_after_vote(scoring, title="Ensemble ranking")
```

### Parallel execution

Speed up ranking by running methods in parallel:

```python
rankings = feature_ranking(X, y, task="classification", n_jobs=-1)
```

### Custom method selection and weights

```python
rankings = feature_ranking(X, y, task="classification", choices=["mi", "f_test", "l1"])
scoring = voting(rankings, weights=[0.2, 0.4, 0.4])
```

### Voting methods

Three aggregation schemes are available:

```python
scoring = voting(rankings, method="reciprocal_rank")  # default: weight * (1/rank)
scoring = voting(rankings, method="borda")             # weight * (n_features - rank)
scoring = voting(rankings, method="exponential")       # weight * exp(-rank / n_features)
```

### Regression

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
df = diabetes.data.merge(diabetes.target, left_index=True, right_index=True)
X, y = get_data(df, target="target")
rankings = feature_ranking(X, y, task="regression")
scoring = voting(rankings)
```

## Ranking Methods

| Key | Method | How it works |
|-----|--------|-------------|
| `rf` | Random Forest | Feature importances from a tuned RandomForest model |
| `xg` | XGBoost | Feature importances from a tuned XGBoost model |
| `mi` | Mutual Information | Statistical dependency between each feature and target |
| `f_test` | ANOVA F-test | Variance-based scoring (f_classif / f_regression) |
| `l1` | L1 Regularization | Regularization path analysis (lasso / logistic L1) |

## Documentation

See the full [API documentation](https://github.com/lhallee/feature-ranker/tree/main/documentation) and [example notebook](https://github.com/lhallee/feature-ranker/blob/main/example_usage.ipynb).

## Development

```bash
git clone https://github.com/lhallee/feature-ranker.git
cd feature-ranker
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

```bibtex
@article{Hallee2023,
  title = {Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life},
  volume = {13},
  ISSN = {2045-2322},
  url = {http://dx.doi.org/10.1038/s41598-023-28965-7},
  DOI = {10.1038/s41598-023-28965-7},
  number = {1},
  journal = {Scientific Reports},
  publisher = {Springer Science and Business Media LLC},
  author = {Hallee, Logan and Khomtchouk, Bohdan B.},
  year = {2023},
  month = feb
}
```

```bibtex
@article{Hallee2023cds,
  title = {cdsBERT - Extending Protein Language Models with Codon Awareness},
  url = {http://dx.doi.org/10.1101/2023.09.15.558027},
  DOI = {10.1101/2023.09.15.558027},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Hallee, Logan and Rafailidis, Nikolaos and Gleghorn, Jason P.},
  year = {2023},
  month = sep
}
```

## License

CC-BY-NC-SA-4.0
