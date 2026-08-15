# featureranker

[![Tests](https://github.com/lhallee/featureranker/actions/workflows/tests.yml/badge.svg)](https://github.com/lhallee/featureranker/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/featureranker)](https://pypi.org/project/featureranker/)

Ensemble feature ranking for any numeric feature matrix: tabular datasets,
transformer embeddings, pooled hidden states, engineered features. Five
methods rank every feature, a typed result object holds the evidence, and
weighted rank voting produces one consensus ordering, for classification and
regression. Unnamed matrices work directly: pass a numpy array and features
get stable generated IDs.

Featured in:

- [Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life](https://www.nature.com/articles/s41598-023-28965-7) (Nature Scientific Reports, 2023)
- [cdsBERT - Extending Protein Language Models with Codon Awareness](https://www.biorxiv.org/content/10.1101/2023.09.15.558027v1.abstract) (bioRxiv, 2023)

## Installation

```bash
pip install featureranker
```

Requires Python >= 3.11.

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from featureranker import feature_ranking, voting, plot_rankings, plot_after_vote

cancer = load_breast_cancer(as_frame=True)
result = feature_ranking(cancer.data, cancer.target, task="classification")

vote_table = voting(result)                # ["feature", "score"], best first
plot_rankings(result, top_n=15)            # per-method ranks as a dot plot
plot_after_vote(vote_table, top_n=15)      # consensus scores
```

`feature_ranking` returns a `RankingResult` with per-method ranking tables,
rank and score matrices, diagnostics, and save/load. Results are
deterministic for a given `random_state` at any `n_jobs`.

## Ranking methods

| Key | Method | Score |
|-----|--------|-------|
| `rf` | Random forest | Impurity importance from a halving-search-tuned forest |
| `xg` | XGBoost | Gain importance from a halving-search-tuned booster |
| `mi` | Mutual information | kNN-estimated dependency with the target |
| `f_test` | ANOVA F-test | Between/within variance ratio |
| `l1` | L1 regularization path | Entry point on the lasso / L1 logistic path |

## Documentation

| Topic | Page |
|---|---|
| Install and first run | [docs/quickstart.md](docs/quickstart.md) |
| The math behind each method and the voting schemes | [docs/algorithms.md](docs/algorithms.md) |
| Speed, the n_jobs core budget, determinism | [docs/performance.md](docs/performance.md) |
| Every signature and exception | [docs/api.md](docs/api.md) |
| Upgrading from v2 | [docs/migration_v2_to_v3.md](docs/migration_v2_to_v3.md) |
| Development and releases | [docs/contributing.md](docs/contributing.md) |

The [example notebook](example_usage.ipynb) walks through classification and
regression end to end.

## Development

```bash
git clone https://github.com/lhallee/featureranker.git
cd featureranker
pip install -e ".[dev]"
pytest
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

MIT
