# Documentation
With help from ChatGPT Python Doc Writer

## [RANKERS.py](https://github.com/lhallee/featureranker/blob/main/src/featureranker/rankers.py)

### `make_ranking(name, cols, importance)`
**Purpose:** Creates a DataFrame ranking features based on their importance scores.

**Usage:** Used to organize ranking results from classification or regression rankings.

**Features:**
* Sorts by score

**Variables:**
* name - string of the methodology in question
* cols - list of features
* importance - list of importance
* returns - pandas dataframe of ranked features in order

### `l1_classification_ranking(X, y, scale, num_alphas, max_iter, norm)` and `l1_regression_ranking(X, y, scale, num_alphas, norm, verbose)`
**Purpose:** Perform feature ranking using L1 regularization for classification and regression tasks.

**Usage:** These functions iteratively apply L1 regularization to determine the order in which features become irrelevant as the regularization strength increases, thus ranking features based on their importance.

**Features:**
* Finds regularization path of l1 regression or classification until all features drop out.
* Results order by nonzero coefficient count. Tie breaks are broken by the higher average coefficient.
* Logistic regression is OvR classification, each class coefficient is averaged and iterations continue until the average is 0.

**Variables:**
* X, y - pandas dataframe / series of features and target/
* scale - float which dictates the scale between the highest and lowest penalization term. Smaller is better but more expensive for regression, larger for classification.
* num_alphas - int which decides how many fits or penalization terms are tried. Larger is better but more expensive.
* norm - bool to normalize the data or not. Recommended, usually faster fits.
* max_iter - int for number of max interations of coordinate descent. Larger is better but more expensive.
* verbose - bool or int, adds progress tracking.
* returns - ranking results as a dataframe.

### `classification_ranking(X, y, cv, num_runs, save, predict, norm, scale, num_alphas, max_iter, choices) and `regression_ranking(..., verbose)`
**Purpose:** Combine multiple feature ranking methods, including Random Forest, XGBoost, Mutual Information, and L1 regularization for both classification and regression.

**Usage:** Provide comprehensive insights into feature importance from various perspectives, enhancing the reliability of the feature selection process.

**Features:**
* Conducts features ranking for each methodology.
* Calls hyper parameter search when using random forests and XGboost.
* Can specify which methods are used with choices.
* Removes NaNs from F-test when the accounted variance is 0.

**Variables:**
* X, y - pandas dataframe / series of features and target/
* cv - integer that decides the k for k-fold cross validation for the hyperparameter search.
* num_runs - integer that decides how many cross validation runs are executed for the hyperparameter search.
* scale - float which dictates the scale between the highest and lowest penalization term. Smaller is better but more expensive for regression, larger for classification.
* num_alphas - int which decides how many fits or penalization terms are tried. Larger is better but more expensive.
* norm - bool to normalize the data or not. Recommended, usually faster fits.
* max_iter - int for number of max interations of coordinate descent. Larger is better but more expensive.
* verbose - bool or int, adds progress tracking.
* returns - a list of tuples with the name and dataframe of ranking results at each index.

### `voting(df, weights=(0.2, 0.2, 0.2, 0.2, 0.2))`
**Purpose:** Implements a voting mechanism to combine different feature ranking methods and calculate a final, aggregated feature ranking.

**Usage:** This function allows for a weighted combination of different feature ranking methods, offering a holistic view of feature importance across various models and techniques.

**Features:**
* Uses the inverse index multiplied by the weights to sum ensemble scores (soft voting).

**Variables:**
* df - pandas dataframe
* weights - tuple of weights associated with each feature ranker in the ensemble.
* returns - list of tuples with feature and final ensemble score.

## [UTILS.py](https://github.com/lhallee/featureranker/blob/main/src/featureranker/utils.py)

### `sanitize_column_names(df)`
**Purpose:** Cleans up DataFrame column names by replacing typical unwanted characters (such as `[]<>{} `) with underscores `_`.

**Usage:** This function is particularly useful when dealing with datasets that have complex or unstructured column names, ensuring compatibility with popular package naming conventions.

**Variables:**
* df - pandas dataframe.
* returns - pandas dataframe.

### `view_data(df)`
**Purpose:** Analyzes a DataFrame for NaN (Not a Number) values and reports the percentage of NaN values in each column.

**Usage:** Essential for preliminary data analysis, this function helps in identifying columns with missing values, guiding the data cleaning and preprocessing steps.

**Variables:**
* df - pandas dataframe.
* prints NaN results.

### `get_data(df, labels, thresh=0.8, columns_to_drop=None)`
**Purpose:** Prepares the dataset for modeling. It separates the labels (target variables), cleans the DataFrame by dropping unnecessary columns, handles missing values based on a threshold, and encodes categorical variables.

**Usage:** Streamlines the data preparation process for machine learning models, ensuring that the input data is clean and formatted correctly.

**Features:**
* Removes unwanted columns.
* Can return more than 1 target for multiple analyses.
* Thresholds NaNs to maintain the row count.

**Variables:**
* df - pandas dataframe.
* labels - list of label column(s).
* thresh - minimum percentage of non NaN value to keep a column.
* columns_to_drop - list of specific columns to drop from the data.
* returns - X, y - feature and target dataframes / series.

### `spearman_scoring_function(y_true, y_pred)`
**Purpose:** Defines a scoring function based on Spearman's rank correlation for regression models.

**Usage:** Used as a custom scoring function in regressive hyperparameter tuning.

**Features:**
*  Uses spearman instead of pearson, not assuming linearity.

**Variables:**
* y_true - list of ground truth values.
* y_pred - list of predicted values.
* returns - spearman r.

### `regression_hyper_param_search(X, y, cv, num_runs, model_params=model_params, save=False)`
**Purpose:** Performs hyperparameter tuning for regression models, using randomized searches over a specified parameter grid.

**Usage:** Gather XGBoost and Random forest hyperparameters for better feature ranking.

**Features:**
* model_params offers a massive variety of hyper parameters for each model.
* Automatically reports correlation metrics and plots CV performance results.
* Increase cv and num_runs to increase the probability of a good model with higher computational cost.
* Option to save associated plots.

**Variables:**
* X, y - pandas dataframe / series of features and target.
* cv - int of how many cross validation splits to do.
* num_runs - int of how many hyperparameter sets to choose from.
* hyper_params - dictionary of dictionaries with many hyperparameter options.
* save - bool to save or not save the associated plots.
* returns - list of dictionaries with best hyperparameters.

### `classification_hyper_param_search(X, y, cv, num_runs, model_params=model_params, save=False)`
**Purpose:** Performs hyperparameter tuning for classification models, using randomized searches over a specified parameter grid.

**Usage:** Gather XGBoost and Random forest hyperparameters for better feature ranking.

**Features:**
* model_params offers a massive variety of hyper parameters for each model.
* Automatically reports correlation metrics and plots CV performance results.
* Increase cv and num_runs to increase the probability of a good model with higher computational cost.
* Option to save associated plots.

**Variables:**
* X, y - pandas dataframe / series of features and target.
* cv - int of how many cross validation splits to do.
* num_runs - int of how many hyperparameter sets to choose from.
* hyper_params - dictionary of dictionaries with many hyperparameter options.
* save - bool to save or not save the associated plots.
* returns - list of dictionaries with best hyperparameters.

## [PLOTS.py](https://github.com/lhallee/featureranker/blob/main/src/featureranker/plots.py)

### `plot_correlations(predictions, labels, model_name, save=False)`
**Purpose:** Generates scatter plots to visualize the relationship between the true values and predictions made by a machine learning model.

**Usage:** Useful for regression tasks to assess the accuracy of model predictions against actual values.

**Features:**
* Displays a line of best fit to illustrate the general trend.
* Annotates the plot with Pearson and Spearman correlation coefficients for additional insights.
* Option to save the plot as an image file.

**Variables:**
* predictions - list of predictions.
* labels - list of ground truth values.
* model_name - string of the associated model.
* save - bool to save or not save the associated plot.

### `plot_confusion_matrix(c_matrix, labels, title='example', save=False)`
**Purpose:** Creates a confusion matrix plot to evaluate the performance of classification models.

**Usage:** Essential for visualizing the accuracy of classification models, highlighting true positives, false positives, true negatives, and false negatives.

**Features:**
* Utilizes color intensity to reflect the magnitude of entries in the matrix.
* Displays actual vs. predicted labels for easy comparison.
* Option to save the plot as an image file.

**Variables:**
* c_matrix - numpy array with confusion matrix results.
* labels - list of associated classification categories.
* title - string for the plot title.
* save - bool to save the plot.


### `plot_after_voting(scoring, title='example', save=False, height_per_feature=0.5, highlight_feature=None)`
**Purpose:** Produces bar charts to showcase ranked features based on aggregate scores.

**Usage**: Ideal for visually representing feature importance rankings, which is crucial in feature selection and understanding model behavior.

**Features:**
* Adjustable height per feature for better visualization of large sets of features.
* Inverted y-axis for a top-down ranking view.
* Option to save the plot as an image file.
* Option to scale the height of the plot per features.
* Option to highlight specific features in yellow instead of blue.

**Variables:**
* scoring - list of tuples with features and ensemble scores.
* title - string for the title of the plot.
* save - bool to save or not save the plot.
* height_per_feature - float scaling factor for the height of the plot per feature.
* highlight_feature - list of feature names to highlight in yellow.


### `plot_rankings(rankings, title, save, heigh_per_feature)`
**Purpose:** Produces overlapping bar charts of each feature ranking method used and their results.

**Usage**: Great for visualizing the difference in various ranking methods.

**Feature:**
* Scalable plot size.
* Different color for each method with opacity.
* Option to save the plot as an image file.

**Variables:**
* rankings - list of tuples output from classification or regression ranking.
* title - string for the title of the plot.
* save - bool to save or not save the plot.
* height_per_feature - float scaling factor for the hegiht of the plot per feature.