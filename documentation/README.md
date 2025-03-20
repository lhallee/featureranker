# Documentation
With help from ChatGPT Python Doc Writer

## [RANKERS.py](https://github.com/lhallee/featureranker/blob/main/src/featureranker/rankers.py)

### `make_ranking(name, cols, importance)`
**Purpose:** Creates a DataFrame ranking features based on their importance scores.

**Usage:** Used to organize ranking results from classification or regression rankings.

**Features:**
* Sorts by score in descending order

**Variables:**
* name - string of the methodology in question
* cols - list of features
* importance - list of importance scores
* returns - pandas dataframe of ranked features in order

### `l1_regression_ranking(X, y)` and `l1_classification_ranking(X, y)`
**Purpose:** Perform feature ranking using L1 regularization for classification and regression tasks.

**Usage:** These functions use L1 regularization to determine the order in which features become irrelevant as the regularization strength increases, thus ranking features based on their importance.

**Features:**
* Finds the regularization path for L1 regression or classification.
* For regression, features are ranked by the maximum alpha at which their coefficients remain non-zero.
* For classification, features are ranked by the maximum C value at which their coefficients remain non-zero.

**Variables:**
* X - pandas dataframe of features
* y - pandas series of target values
* returns - ranking results as a dataframe with columns 'L1' (feature names) and 'Score'

### `feature_ranking(X, y, task="classification", choices=None, save=False, save_path=None, **kwargs)`
**Purpose:** Combines multiple feature ranking methods, including Random Forest, XGBoost, Mutual Information, F-test, and L1 regularization for both classification and regression.

**Usage:** Provides comprehensive insights into feature importance from various perspectives, enhancing the reliability of the feature selection process.

**Features:**
* Conducts feature ranking for each selected methodology.
* Performs hyperparameter search when using Random Forests and XGBoost.
* Allows specifying which methods to use with the choices parameter.
* Supports saving results to a file.

**Variables:**
* X - pandas dataframe of features
* y - pandas series of target values
* task - string indicating "classification" or "regression"
* choices - list of methods to use (options: "rf", "xg", "mi", "f_test", "l1")
* save - bool indicating whether to save the results
* save_path - path to save the results, if None a default name is used
* **kwargs - additional parameters passed to the hyperparameter search functions
* returns - a list of tuples with the name and dataframe of ranking results at each index

### `voting(rankings, weights=None, save=False, save_path=None)`
**Purpose:** Implements a voting mechanism to combine different feature ranking methods and calculate a final, aggregated feature ranking.

**Usage:** This function allows for a weighted combination of different feature ranking methods, offering a holistic view of feature importance across various models and techniques.

**Features:**
* Combines rankings from multiple methods into a single score.
* Supports custom weighting of different ranking methods.
* Option to save the final ranking.

**Variables:**
* rankings - list of tuples with ranking method names and their DataFrames
* weights - list of weights for each ranking method (defaults to equal weights)
* save - bool indicating whether to save the results
* save_path - path to save the results
* returns - pandas DataFrame with features and their ensemble scores

## [UTILS.py](https://github.com/lhallee/featureranker/blob/main/src/featureranker/utils.py)

### `sanitize_column_names(df)`
**Purpose:** Cleans up DataFrame column names by replacing unwanted characters with underscores.

**Usage:** This function is useful when dealing with datasets that have complex or unstructured column names.

**Variables:**
* df - pandas dataframe
* returns - pandas dataframe with sanitized column names

### `view_data(df)`
**Purpose:** Analyzes a DataFrame for NaN values and reports their percentage in each column.

**Usage:** Essential for preliminary data analysis to identify columns with missing values.

**Variables:**
* df - pandas dataframe
* prints the percentage of NaN values in each column

### `get_data(df, target, thresh=0.8, columns_to_drop=None, n_rows=None)`
**Purpose:** Prepares the dataset for modeling by cleaning and encoding features.

**Usage:** Streamlines the data preparation process for machine learning models.

**Features:**
* Removes specified columns and those with too many missing values.
* Encodes categorical columns.
* Can shuffle and sample a specified number of rows.
* Removes constant columns.

**Variables:**
* df - pandas dataframe
* target - name of the target column
* thresh - minimum percentage of non-NaN values to keep a column
* columns_to_drop - list of specific columns to drop
* n_rows - if specified, sample this many rows from the dataset
* returns - X, y tuple with feature matrix and target vector

### `spearman_scoring_function(y_true, y_pred)`
**Purpose:** Defines a scoring function based on Spearman's rank correlation for regression models.

**Usage:** Used as a custom scoring function in regression hyperparameter tuning.

**Variables:**
* y_true - array of ground truth values
* y_pred - array of predicted values
* returns - Spearman correlation coefficient

### `regression_hyper_param_search(X, y, model_name, cv=3, n_iter=5, verbose=2, n_jobs=-1, model_params=model_params, save=False, predict=True)`
**Purpose:** Performs hyperparameter optimization for regression models using Bayesian optimization.

**Usage:** Helps find optimal hyperparameters for regression models (RandomForest and XGBoost).

**Features:**
* Uses Bayesian optimization for efficient hyperparameter search.
* Supports cross-validation for robust evaluation.
* Optional prediction visualization and saving of results.

**Variables:**
* X - pandas dataframe of features
* y - pandas series of target values
* model_name - name of the model ("RandomForest" or "XGBoost")
* cv - number of cross-validation folds
* n_iter - number of iterations for the search
* verbose - verbosity level
* n_jobs - number of parallel jobs
* model_params - dictionary of model parameters
* save - whether to save the results
* predict - whether to make and visualize predictions
* returns - dictionary of best hyperparameters

### `classification_hyper_param_search(X, y, model_name, cv=3, n_iter=5, verbose=2, n_jobs=-1, model_params=model_params, save=False, predict=True)`
**Purpose:** Performs hyperparameter optimization for classification models using Bayesian optimization.

**Usage:** Helps find optimal hyperparameters for classification models (RandomForest and XGBoost).

**Features:**
* Uses Bayesian optimization for efficient hyperparameter search.
* Supports cross-validation for robust evaluation.
* Optional confusion matrix visualization and saving of results.

**Variables:**
* X - pandas dataframe of features
* y - pandas series of target values
* model_name - name of the model ("RandomForest" or "XGBoost")
* cv - number of cross-validation folds
* n_iter - number of iterations for the search
* verbose - verbosity level
* n_jobs - number of parallel jobs
* model_params - dictionary of model parameters
* save - whether to save the results
* predict - whether to make and visualize predictions
* returns - dictionary of best hyperparameters

## [PLOTS.py](https://github.com/lhallee/featureranker/blob/main/src/featureranker/plots.py)

### `plot_correlations(predictions, labels, model_name, save=False)`
**Purpose:** Generates scatter plots to visualize the relationship between true values and predictions.

**Usage:** Useful for evaluating regression model performance.

**Features:**
* Displays a line of best fit.
* Shows Pearson and Spearman correlation coefficients and R² score.
* Option to save the plot.

**Variables:**
* predictions - array of predicted values
* labels - array of true values
* model_name - name of the model for the plot title
* save - whether to save the plot

### `plot_confusion_matrix(c_matrix, labels, title='example', save=False)`
**Purpose:** Creates a confusion matrix visualization for classification model evaluation.

**Usage:** Essential for visualizing classification model performance.

**Features:**
* Color-coded matrix cells for better visualization.
* Displays actual counts in each cell.
* Option to save the plot.

**Variables:**
* c_matrix - confusion matrix array
* labels - classification category labels
* title - plot title
* save - whether to save the plot

### `plot_after_vote(scoring, title='example', save=False, height_per_feature=0.25, highlight_feature=None)`
**Purpose:** Produces bar charts to visualize ranked features based on aggregate scores.

**Usage:** Ideal for displaying feature importance rankings after voting.

**Features:**
* Adjustable plot height based on the number of features.
* Inverted y-axis for top-down ranking view.
* Option to highlight specific features.
* Option to save the plot.

**Variables:**
* scoring - DataFrame with 'Feature' and 'Score' columns
* title - plot title
* save - whether to save the plot
* height_per_feature - scaling factor for plot height
* highlight_feature - feature name to highlight in yellow

### `plot_rankings(rankings, title='example', save=False, height_per_feature=0.25)`
**Purpose:** Produces overlapping bar charts for multiple feature ranking methods.

**Usage:** Great for comparing different ranking methods side by side.

**Features:**
* Different color for each method with transparency.
* Includes a legend for each ranking method.
* Adjustable plot size based on the number of features.
* Option to save the plot.

**Variables:**
* rankings - list of tuples with ranking method names and DataFrames
* title - plot title
* save - whether to save the plot
* height_per_feature - scaling factor for plot height

## [CLUSTERING.py](https://github.com/lhallee/featureranker/blob/main/src/featureranker/clustering.py)

### `random_cluster_generator(n_samples=1000, n_features=2, n_centers=3, std=1.0)`
**Purpose:** Generates random clusters for testing and demonstration purposes.

**Usage:** Useful for creating synthetic data to test clustering algorithms.

**Variables:**
* n_samples - number of samples to generate
* n_features - number of features
* n_centers - number of cluster centers
* std - standard deviation of the clusters
* returns - numpy array with generated clusters

### `get_inertia(X, k)`
**Purpose:** Calculates the inertia (within-cluster sum of squares) for K-means clustering.

**Usage:** Used as a helper function for determining optimal cluster numbers.

**Variables:**
* X - feature matrix
* k - number of clusters
* returns - inertia value

### `optimal_k_w_elbow(X, max_k=10)`
**Purpose:** Determines the optimal number of clusters using the elbow method.

**Usage:** Helps in deciding the optimal number of clusters for K-means.

**Features:**
* Implements the elbow method by finding the point of maximum distance from a line.

**Variables:**
* X - feature matrix
* max_k - maximum number of clusters to try
* returns - optimal number of clusters

### `get_kmean_metrics(X, k)`
**Purpose:** Calculates both inertia and silhouette score for K-means clustering.

**Usage:** Provides multiple metrics to evaluate clustering quality.

**Variables:**
* X - feature matrix
* k - number of clusters
* returns - tuple with inertia and silhouette score

### `optimal_k_w_both(X, max_k=10)`
**Purpose:** Determines the optimal number of clusters using both the elbow method and silhouette scores.

**Usage:** Provides a more robust way to find the optimal number of clusters.

**Features:**
* Combines the elbow method with silhouette scores for better cluster number determination.

**Variables:**
* X - feature matrix
* max_k - maximum number of clusters to try
* returns - optimal number of clusters