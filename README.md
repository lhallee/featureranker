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
rankings = regression_ranking(X, y, predict=False)
scoring = voting(rankings)
plot_rankings(rankings, title='Regression example all methods')
plot_after_vote(scoring, title='Regression example full ensemble')
```
![image](https://github.com/lhallee/featureranker/assets/72926928/a95c8ac9-11b5-45df-827f-0be1255c82ea)
![image](https://github.com/lhallee/featureranker/assets/72926928/710ed10e-eed5-4f0e-b9f8-997c7fb0de8b)

Classification example (breast cancer dataset)
```
cancer = load_breast_cancer(as_frame=True)
df = cancer.data.merge(cancer.target, left_index=True, right_index=True)
view_data(df)
X, y = get_data(df, labels='target')
rankings = classification_ranking(X, y, predict=False)
scoring = voting(rankings)
plot_rankings(rankings, title='Classification example all methods')
plot_after_vote(scoring, title='Classification example full ensemble')
```
![image](https://github.com/lhallee/featureranker/assets/72926928/fbb1308f-118f-4db2-a5a4-9c65d510fbc3)
![image](https://github.com/lhallee/featureranker/assets/72926928/88373375-18a3-4c82-99b2-1aec7b79aaa4)

### [More examples](https://github.com/lhallee/featureranker/tree/main/examples)

## [Documentation](https://github.com/lhallee/featureranker/tree/main/documentation)
See documentation via the link above for more details

## ISSUES WITH GOOGLE COLAB
The numpy / linux build on Google Colab does not always work when installing featureranker on collab.
**Simply upgrade numpy and restart the session to fix featureranker.**

## Citation
Please cite 
_Hallee, L., Khomtchouk, B.B. Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life. Sci Rep 13, 2088 (2023).
https://doi.org/10.1038/s41598-023-28965-7_

and

_Logan Hallee, Nikolaos Rafailidis, Jason P. Gleghorn
bioRxiv 2023.09.15.558027; doi: https://doi.org/10.1101/2023.09.15.558027_

## News
* 1/22/2023: Version 1.1.0 is released with faster solvers, many more settings, and more plots. 1.1.1 fixes some bugs.
* 1/3/2023: Version 1.0.2 is released with added clustering capabilities and better automatic plots.
* 11/10/2023: Version 1.0.1 is published in PyPI under featureranker.
* 11/9/2023: Version 1.0.0 of the package is published for testing on TestPyPI.
* 11/8/2023: Various utility helpers and plot functions are added for ease of use. The proper l1 penalty constant is now found automatically. The automatic hyperparameter search also returns the best metrics found via the methodologies.
* 11/7/2023: Recursive feature extraction is replaced with ANOVA F-scores due to its ability to rank based on modeled variance.
* 10/15/2023: A separate classification and regression version are developed for more reliable results. Logistic regression (OvR) with an l1 penalty takes the place of lasso for classification.
* 9/17/2023: The feature ranker is now a proper ensemble, with a custom soft voting scheme. XGboost, recursive feature elimination, and mutual information are also leveraged. The ensemble is used to unify the results of the previous papers in the cdsBERT paper.
* 2/6/2023: The preliminary work makes its way into Nature Scientific Reports!
* 7/21/2022: A preliminary version of this feature ranker leveraging lasso and random forests is published in BioRxiv for phylogenetic and organelle prediction.

