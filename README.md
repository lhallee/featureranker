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
### [Example use](https://github.com/lhallee/featureranker/blob/main/example_usage.ipynb)

## [Documentation](https://github.com/lhallee/featureranker/tree/main/documentation)

## ISSUES WITH GOOGLE COLAB
The numpy / linux build on Google Colab does not always work when installing featureranker on collab.
**Simply upgrade numpy and restart the session to fix featureranker.**

## Citation
Please cite 
```
@article{Hallee2023,
  title = {Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life},
  volume = {13},
  ISSN = {2045-2322},
  url = {http://dx.doi.org/10.1038/s41598-023-28965-7},
  DOI = {10.1038/s41598-023-28965-7},
  number = {1},
  journal = {Scientific Reports},
  publisher = {Springer Science and Business Media LLC},
  author = {Hallee,  Logan and Khomtchouk,  Bohdan B.},
  year = {2023},
  month = feb 
}
```
```
@article{Hallee2023,
  title = {cdsBERT - Extending Protein Language Models with Codon Awareness},
  url = {http://dx.doi.org/10.1101/2023.09.15.558027},
  DOI = {10.1101/2023.09.15.558027},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Hallee,  Logan and Rafailidis,  Nikolaos and Gleghorn,  Jason P.},
  year = {2023},
  month = sep 
}
```

## News
* 3/20/2025: Version 1.3.0 is released with improved runtime, documentation, and examples.
* 10/22/2024: Versions 1.2.0 - 1.2.2 are released with improvements and bug fixes.
* 1/22/2024: Version 1.1.0 is released with faster solvers, many more settings, and more plots. 1.1.1 fixes some bugs.
* 1/3/2024: Version 1.0.2 is released with added clustering capabilities and better automatic plots.
* 11/10/2023: Version 1.0.1 is published in PyPI under featureranker.
* 11/9/2023: Version 1.0.0 of the package is published for testing on TestPyPI.
* 11/8/2023: Various utility helpers and plot functions are added for ease of use. The proper l1 penalty constant is now found automatically. The automatic hyperparameter search also returns the best metrics found via the methodologies.
* 11/7/2023: Recursive feature extraction is replaced with ANOVA F-scores due to its ability to rank based on modeled variance.
* 10/15/2023: A separate classification and regression version are developed for more reliable results. Logistic regression (OvR) with an l1 penalty takes the place of lasso for classification.
* 9/17/2023: The feature ranker is now a proper ensemble, with a custom soft voting scheme. XGboost, recursive feature elimination, and mutual information are also leveraged. The ensemble is used to unify the results of the previous papers in the cdsBERT paper.
* 2/6/2023: The preliminary work makes its way into Nature Scientific Reports!
* 7/21/2022: A preliminary version of this feature ranker leveraging lasso and random forests is published in BioRxiv for phylogenetic and organelle prediction.

