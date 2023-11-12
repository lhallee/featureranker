# FEATURE RANKER
featureranker is a lightweight Python package for the feature ranking ensemble developed by Logan Hallee, featured in the following works:

[Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life](https://www.nature.com/articles/s41598-023-28965-7)

[cdsBERT - Extending Protein Language Models with Codon Awareness](https://www.biorxiv.org/content/10.1101/2023.09.15.558027v1.abstract)

[Exploring Phylogenetic Classification and Further Applications of Codon Usage Frequencies](https://www.biorxiv.org/content/10.1101/2022.07.20.500846v1.abstract)

The ensemble utilizes l1 penalization, random forests, extreme gradient boosting, ANOVA F values, and mutual information to effectively rank the importance of features for regression and classification tasks. Scoring lists are concatenated with a weighted voting scheme.

## Usage

Install
```
!pip install featureranker
```
Imports

```
from featureranker.utils import *
from featureranker.plots import *
from featureranker.rankers import *

import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
```
Regression example (diabetes dataset)
```
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

<img src="https://github.com/lhallee/featureranker/assets/72926928/8b8a2237-d5fb-4c72-a684-3ddfdccaa5bd" width="500"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/ef258433-aa56-447b-8847-2d391be6a941" width="500"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/912fdc22-7652-474d-9668-fe4da75b2473" width="500"/>

Classification example (breast cancer dataset)
```
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

<img src="https://github.com/lhallee/featureranker/assets/72926928/7c61cfa6-7bd3-40f0-a319-7d00c2e7e743" width="500"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/088ed7ea-098e-4ef7-ab26-d5f1dff88106" width="500"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/63100c6e-2b79-496d-9d3c-640593ccc1d7" width="500"/>


## [Documentation](https://github.com/lhallee/featureranker/tree/main/documentation)
See documentation via the link above for more details

## Citation
Please cite 
_Hallee, L., Khomtchouk, B.B. Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life. Sci Rep 13, 2088 (2023).
https://doi.org/10.1038/s41598-023-28965-7_

and

_Logan Hallee, Nikolaos Rafailidis, Jason P. Gleghorn
bioRxiv 2023.09.15.558027; doi: https://doi.org/10.1101/2023.09.15.558027_

## News
* 7/21/2022: A preliminary version of this feature ranker leveraging lasso and random forests is published in BioRxiv for phylogenetic and organelle prediction.
* 2/6/2023: The preliminary work makes its way into Nature Scientific Reports!
* 9/17/2023: The feature ranker is now a proper ensemble, with a custom soft voting scheme. XGboost, recursive feature elimination, and mutual information are also leveraged. The ensemble is used to unify the results of the previous papers in the cdsBERT paper.
* 10/15/2023: A separate classification and regression version are developed for more reliable results. Logistic regression (OvR) with an l1 penalty takes the place of lasso for classification.
* 11/7/2023: Recursive feature extraction is replaced with ANOVA F-scores due to its ability to rank based on modeled variance.
* 11/8/2023: Various utility helpers and plot functions are added for ease of use. The proper l1 penalty constant is now found automatically. The automatic hyperparameter search also returns the best metrics found via the methodologies.
* 11/9/2023: Version 1.0.0 of the package is published for testing on TestPyPI.
* 11/10/2023: Version 1.0.1 is published in PyPI under featureranker.

