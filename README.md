# FEATURE RANKER
featureranker is a lightweight framework for the feature ranking ensemble developed by Logan Hallee, featured in the following works:

[Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life](https://www.nature.com/articles/s41598-023-28965-7)

[cdsBERT - Extending Protein Language Models with Codon Awareness](https://www.biorxiv.org/content/10.1101/2023.09.15.558027v1.abstract)

[Exploring Phylogenetic Classification and Further Applications of Codon Usage Frequencies](https://www.biorxiv.org/content/10.1101/2022.07.20.500846v1.abstract)

Install
```
!pip install featureranker
```
Example usage (found in examples)
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

<img src="https://github.com/lhallee/featureranker/assets/72926928/088ed7ea-098e-4ef7-ab26-d5f1dff88106" width="400"/>

<img src="https://github.com/lhallee/featureranker/assets/72926928/63100c6e-2b79-496d-9d3c-640593ccc1d7" width="400"/>

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


# Documentation
With help from ChatGPT Python Doc Writer
## RANKERS.py

### `make_ranking(name, cols, importance)`
**Purpose:** Creates a DataFrame ranking features based on their importance scores.  
**Usage:** This function is versatile and can be used to rank features from different models or methods by providing the feature names and their corresponding importance scores.

### `find_alpha_step(X, y, alpha_iter=100, init_step=1e-20)` and `find_c_step(X, y, init_step=1e20)`
**Purpose:** These functions find the appropriate step size for alpha (Lasso regression) and C (Logistic regression) in L1 regularization.  
**Usage:** They help in fine-tuning the L1 regularization process, ensuring more precise feature selection.

### `l1_classification_ranking(X, y)` and `l1_regression_ranking(X, y)`
**Purpose:** Perform feature ranking using L1 regularization for classification and regression tasks.  
**Usage:** These functions iteratively apply L1 regularization to determine the order in which features become irrelevant as the regularization strength increases, thus ranking features based on their importance.

### `classification_ranking(X, y, rf_hyper, xb_hyper)` and `regression_ranking(X, y, rf_hyper, xb_hyper)`
**Purpose:** Combine multiple feature ranking methods, including Random Forest, XGBoost, Mutual Information, and L1 regularization for both classification and regression.  
**Usage:** Provide comprehensive insights into feature importance from various perspectives, enhancing the reliability of the feature selection process.

### `voting(df, weights=(0.2, 0.2, 0.2, 0.2, 0.2))`
**Purpose:** Implements a voting mechanism to combine different feature ranking methods and calculate a final, aggregated feature ranking.  
**Usage:** This function allows for a weighted combination of different feature ranking methods, offering a holistic view of feature importance across various models and techniques.

## UTILS.py

### `sanitize_column_names(df)`
**Purpose:** Cleans up DataFrame column names by replacing typical unwanted characters (such as `[]<>{} `) with underscores `_`.  
**Usage:** This function is particularly useful when dealing with datasets that have complex or unstructured column names, ensuring compatibility with Python’s naming conventions.

### `view_data(df)`
**Purpose:** Analyzes a DataFrame for NaN (Not a Number) values and reports the percentage of NaN values in each column.  
**Usage:** Essential for preliminary data analysis, this function helps in identifying columns with missing values, guiding the data cleaning and preprocessing steps.

### `get_data(df, labels, thresh=0.8, columns_to_drop=None)`
**Purpose:** Prepares the dataset for modeling. It separates the labels (target variables), cleans the DataFrame by dropping unnecessary columns, handles missing values based on a threshold, and encodes categorical variables.  
**Usage:** Streamlines the data preparation process for machine learning models, ensuring that the input data is clean and formatted correctly.

### `spearman_scoring_function(y_true, y_pred)`
**Purpose:** Defines a scoring function based on Spearman's rank correlation for regression models.  
**Usage:** Used as a custom scoring function in hyperparameter tuning, particularly useful for assessing non-linear relationships in regression tasks.

### `regression_hyper_param_search(X, y, cv, num_runs, model_params=model_params, save=False)`
**Purpose:** Performs hyperparameter tuning for regression models, using randomized searches over a specified parameter grid.  
**Usage:** Essential for optimizing regression models, this function helps in finding the best model parameters, and optionally saves the correlation plots between predictions and actual values.

### `classification_hyper_param_search(X, y, cv, num_runs, model_params=model_params, save=False)`
**Purpose:** Similar to the regression hyperparameter search, but tailored for classification models.  
**Usage:** Facilitates the optimization of classification models, evaluates model performance, and visualizes results through confusion matrices.

## PLOTS.py

### `plot_correlations(predictions, labels, model_name, save=False)`
**Purpose:** Generates scatter plots to visualize the relationship between the true values and predictions made by a machine learning model.  
**Features:**
- Displays a line of best fit to illustrate the general trend.
- Annotates the plot with Pearson and Spearman correlation coefficients for additional insights.
- Option to save the plot as an image file.  
**Usage:** Useful for regression tasks to assess the accuracy of model predictions against actual values.

### `plot_confusion_matrix(c_matrix, labels, title='example', save=False)`
**Purpose:** Creates a confusion matrix plot to evaluate the performance of classification models.  
**Features:**
- Utilizes color intensity to reflect the magnitude of entries in the matrix.
- Displays actual vs. predicted labels for easy comparison.
- Option to save the plot as an image file.  
**Usage:** Essential for visualizing the accuracy of classification models, highlighting true positives, false positives, true negatives, and false negatives.

### `plot_ranking(scoring, title='example', save=False, height_per_feature=0.5)`
**Purpose:** Produces bar charts to showcase ranked features based on aggregate scores.
**Features:**
- Adjustable height per feature for better visualization of large sets of features.
- Inverted y-axis for a top-down ranking view.
- Option to save the plot as an image file.
**Usage**: Ideal for visually representing feature importance rankings, which is crucial in feature selection and understanding model behavior.






