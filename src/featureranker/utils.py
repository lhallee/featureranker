import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import spearmanr
from skopt import BayesSearchCV

from featureranker.plots import plot_confusion_matrix, plot_correlations


model_params = {
    "classification": {
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "params": {
                "max_depth": (3, 50),
                "min_child_weight": (1, 10),
                "gamma": (0.0, 0.5),
                "subsample": (0.5, 1.0),
                "colsample_bytree": (0.5, 1.0),
                "learning_rate": (0.01, 0.5, "log-uniform"),
                "n_estimators": (100, 1000),
                "reg_alpha": (0.1, 100, "log-uniform"),
                "reg_lambda": (0.1, 100, "log-uniform"),
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": (10, 1000),
                "max_features": ["sqrt", "log2", None],
                "max_depth": (10, 100),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 4),
                "bootstrap": [True, False],
            },
        },
    },
    "regression": {
        "XGBoost": {
            "model": XGBRegressor(),
            "params": {
                "max_depth": (3, 50),
                "min_child_weight": (1, 10),
                "gamma": (0.0, 0.5),
                "subsample": (0.5, 1.0),
                "colsample_bytree": (0.5, 1.0),
                "learning_rate": (0.01, 0.5, "log-uniform"),
                "n_estimators": (100, 1000),
                "reg_alpha": (0.1, 100, "log-uniform"),
                "reg_lambda": (0.1, 100, "log-uniform"),
            },
        },
        "RandomForest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": (10, 1000),
                "max_features": ["sqrt", "log2", None],
                "max_depth": (10, 100),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 4),
                "bootstrap": [True, False],
            },
        },
    },
}


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace unwanted characters in column names with underscores.

    Parameters:
        df (pd.DataFrame): DataFrame with columns to sanitize.

    Returns:
        pd.DataFrame: DataFrame with sanitized column names.
    """
    df.columns = [re.sub(r"[^\w]", "_", col) for col in df.columns]
    return df


def view_data(df: pd.DataFrame) -> None:
    """
    Display the percentage of NaN values in each column of the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        None
    """
    has_nans = False
    for column in df.columns:
        nan_count = df[column].isna().sum()
        nan_percentage = round(nan_count / len(df) * 100, 1)
        if nan_percentage > 0:
            has_nans = True
            print(f"The column {column} has {nan_percentage}% NaN values.")
    if not has_nans:
        print("There are no NaN values in the dataset.")


def get_data(
    df: pd.DataFrame,
    target: str,
    thresh: float = 0.8,
    columns_to_drop: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the dataset by cleaning and encoding features.

    Features added:
        1. Removes columns with constant values and prints their names.
        2. Shuffles the dataset and returns a specified number of rows if `n_rows` is provided.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        target (str): The target column name.
        thresh (float): Threshold for dropping columns with missing values (default is 0.8).
        columns_to_drop (List[str], optional): List of columns to drop (default is None).
        n_rows (int, optional): If specified, shuffle and return this number of rows (default is None).

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix and target vector.

    Raises:
        ValueError: If specified columns to drop are not in the DataFrame.
        ValueError: If the target column is not found in the DataFrame.
        ValueError: If `n_rows` is greater than the available rows after cleaning.
    """
    # Drop specified columns if any
    if columns_to_drop:
        missing_cols = set(columns_to_drop) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")

    # Ensure target column exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    y = df[target]
    df_clean = df.drop(columns=[target])

    # Drop columns with missing values below the threshold
    threshold = thresh * len(df_clean)
    df_clean = df_clean.dropna(axis=1, thresh=threshold)
    print(f"Column count after dropping those with > {thresh*100}% missing values: {len(df_clean.columns)}")

    # Combine features and target to drop rows with any remaining NaNs
    combined = pd.concat([df_clean, y], axis=1)
    combined_clean = combined.dropna()
    df_clean = combined_clean[df_clean.columns]
    y = combined_clean[target]

    # Shuffle and sample rows if n_rows is specified
    if n_rows is not None:
        total_rows = len(df_clean)
        if n_rows > total_rows:
            raise ValueError(f"Requested number of rows ({n_rows}) exceeds available rows ({total_rows}).")
        combined_clean = combined_clean.sample(n=n_rows, random_state=42).reset_index(drop=True)
        df_clean = combined_clean[df_clean.columns]
        y = combined_clean[target].reset_index(drop=True)
        print(f"Shuffled and sampled {n_rows} rows from the dataset.")

    # Remove constant columns
    constant_columns = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
    if constant_columns:
        df_clean = df_clean.drop(columns=constant_columns)
        print(f"Removed constant columns: {constant_columns}")
    else:
        print("No constant columns found.")

    # Encode categorical columns
    le = LabelEncoder()
    columns_to_encode = df_clean.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    for column in columns_to_encode:
        df_clean[column] = le.fit_transform(df_clean[column])
        print(f"Encoded column: {column}")

    X = df_clean

    # Convert boolean target to integer if necessary
    if y.dtype == "bool":
        y = y.astype(int)
        print("Converted boolean target to integer.")

    return X, y


def spearman_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Custom scoring function for Spearman correlation.

    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Spearman correlation coefficient.
    """
    return spearmanr(y_true, y_pred)[0]


def regression_hyper_param_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv: int = 3,
    n_iter: int = 5,
    verbose: int = 0,
    model_params: Dict = model_params,
    save: bool = False,
    predict: bool = True,
) -> Dict:
    """
    Perform hyperparameter optimization for regression models using Bayesian optimization.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        model_name (str): Name of the regression model.
        cv (int): Number of cross-validation folds.
        n_iter (int): Number of iterations for the search.
        verbose (int): Verbosity level.

    Returns:
        Dict: Best hyperparameters found.
    """
    mp = model_params["regression"][model_name]
    opt = BayesSearchCV(
        estimator=mp["model"],
        search_spaces=mp["params"],
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=verbose,
        random_state=42,
    )
    opt.fit(X, y)
    if predict:
        predictions = cross_val_predict(opt.best_estimator_, X, y, cv=cv, n_jobs=-1)
        plot_correlations(predictions, y, model_name, save=save)
    return opt.best_params_


def classification_hyper_param_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv: int = 3,
    n_iter: int = 5,
    verbose: int = 0,
    model_params: Dict = model_params,
    save: bool = False,
    predict: bool = True,
) -> Dict:
    """
    Perform hyperparameter optimization for classification models using Bayesian optimization.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        model_name (str): Name of the classification model.
        cv (int): Number of cross-validation folds.
        n_iter (int): Number of iterations for the search.
        verbose (int): Verbosity level.

    Returns:
        Dict: Best hyperparameters found.
    """
    mp = model_params["classification"][model_name]
    opt = BayesSearchCV(
        estimator=mp["model"],
        search_spaces=mp["params"],
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=verbose,
        random_state=42,
    )
    opt.fit(X, y)
    if predict:
        predictions = cross_val_predict(opt.best_estimator_, X, y, cv=cv, n_jobs=-1)
        acc = accuracy_score(y, predictions)
        cm = confusion_matrix(y, predictions)
        plot_confusion_matrix(
            cm,
            title=f"Confusion matrix for {model_name} with {round(acc, 3)} accuracy",
            labels=np.unique(y),
            save=save,
        )
    return opt.best_params_
