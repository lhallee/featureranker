# FEATURE RANKER
featureranker is a lightweight Python package for the feature ranking ensemble developed by Logan Hallee, featured in the following works:

[Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life](https://www.nature.com/articles/s41598-023-28965-7)

[cdsBERT - Extending Protein Language Models with Codon Awareness](https://www.biorxiv.org/content/10.1101/2023.09.15.558027v1.abstract)

[Exploring Phylogenetic Classification and Further Applications of Codon Usage Frequencies](https://www.biorxiv.org/content/10.1101/2022.07.20.500846v1.abstract)

## Usage

Install
```
!pip install featureranker
```
Examples
```
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
from featureranker.utils import *
from featureranker.plots import *
from featureranker.rankers import *
# Regression example
diabetes = load_diabetes(as_frame=True)
df = diabetes.data.merge(diabetes.target, left_index=True, right_index=True)
view_data(df)
X, y = get_data(df, labels='target')
hypers = regression_hyper_param_search(X, y, 3, 5)
xb_hypers = hypers[0]['best_params']
rf_hypers = hypers[1]['best_params']
ranking = regression_ranking(X, y, rf_hypers, xb_hypers)
scoring = voting(ranking)
plot_ranking(scoring, title='Regression example')
```

<img src="https://github.com/lhallee/featureranker/assets/72926928/7c61cfa6-7bd3-40f0-a319-7d00c2e7e743" width="400"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/ef258433-aa56-447b-8847-2d391be6a941" width="400"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/912fdc22-7652-474d-9668-fe4da75b2473" width="400"/>

```
# Classification example
cancer = load_breast_cancer(as_frame=True)
df = cancer.data.merge(cancer.target, left_index=True, right_index=True)
view_data(df)
X, y = get_data(df, labels='target')
hypers = classification_hyper_param_search(X, y, 3, 5)
xb_hypers = hypers[0]['best_params']
rf_hypers = hypers[1]['best_params']
ranking = classification_ranking(X, y, rf_hypers, xb_hypers)
scoring = voting(ranking)
plot_ranking(scoring, title='Classification example')
```

<img src="https://github.com/lhallee/featureranker/assets/72926928/7c61cfa6-7bd3-40f0-a319-7d00c2e7e743" width="400"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/088ed7ea-098e-4ef7-ab26-d5f1dff88106" width="400"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/63100c6e-2b79-496d-9d3c-640593ccc1d7" width="400"/>


## [Documentation](https://github.com/lhallee/featureranker/tree/main/documentation)
See documentation for more details

## Citation
Please cite 
_Hallee, L., Khomtchouk, B.B. Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life. Sci Rep 13, 2088 (2023).
https://doi.org/10.1038/s41598-023-28965-7_

and

_Logan Hallee, Nikolaos Rafailidis, Jason P. Gleghorn
bioRxiv 2023.09.15.558027; doi: https://doi.org/10.1101/2023.09.15.558027_


