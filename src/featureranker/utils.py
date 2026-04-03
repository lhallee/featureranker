import logging
import re

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


def _get_search_config(model_name: str, task: str) -> tuple:
    """Return (estimator, param_distributions) for RandomizedSearchCV."""
    rf_params = {
        "n_estimators": randint(10, 1000),
        "max_features": ["sqrt", "log2", None],
        "max_depth": randint(10, 100),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 4),
        "bootstrap": [True, False],
    }
    xgb_params = {
        "max_depth": randint(3, 50),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0.0, 0.5),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "learning_rate": uniform(0.01, 0.49),
        "n_estimators": randint(100, 1000),
        "reg_alpha": uniform(0.1, 99.9),
        "reg_lambda": uniform(0.1, 99.9),
    }

    if model_name == "RandomForest":
        if task == "classification":
            return RandomForestClassifier(random_state=42), rf_params
        return RandomForestRegressor(random_state=42), rf_params
    elif model_name == "XGBoost":
        if task == "classification":
            return XGBClassifier(eval_metric="logloss", random_state=42), xgb_params
        return XGBRegressor(random_state=42), xgb_params
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'RandomForest' or 'XGBoost'.")


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-word characters in column names with underscores."""
    df = df.copy()
    df.columns = [re.sub(r"[^\w]", "_", col) for col in df.columns]
    return df


def view_data(df: pd.DataFrame) -> pd.Series:
    """Return percentage of NaN values per column. Only includes columns with NaNs."""
    nan_pct = (df.isna().sum() / len(df) * 100).round(1)
    nan_pct = nan_pct[nan_pct > 0]
    if nan_pct.empty:
        logger.info("No NaN values in the dataset.")
    else:
        for col, pct in nan_pct.items():
            logger.info("Column %s has %.1f%% NaN values.", col, pct)
    return nan_pct


def get_data(
    df: pd.DataFrame,
    target: str,
    thresh: float = 0.8,
    columns_to_drop: list[str] | None = None,
    n_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare dataset by cleaning and encoding features.

    Steps:
        1. Drop specified columns
        2. Drop columns with more than (1-thresh)*100% missing values
        3. Drop rows with remaining NaNs
        4. Optionally shuffle and sample n_rows
        5. Remove constant columns
        6. Label-encode categorical/string/bool columns
    """
    df = df.copy()

    if columns_to_drop:
        missing_cols = set(columns_to_drop) - set(df.columns)
        assert not missing_cols, f"Columns not found in DataFrame: {missing_cols}"
        df = df.drop(columns=columns_to_drop)
        logger.info("Dropped columns: %s", columns_to_drop)

    assert target in df.columns, f"Target column '{target}' not found in DataFrame."

    y = df[target]
    df_clean = df.drop(columns=[target])

    threshold = int(thresh * len(df_clean))
    df_clean = df_clean.dropna(axis=1, thresh=threshold)
    logger.info(
        "Column count after dropping those with >%.0f%% missing: %d",
        (1 - thresh) * 100,
        len(df_clean.columns),
    )

    combined = pd.concat([df_clean, y], axis=1).dropna()
    df_clean = combined[df_clean.columns]
    y = combined[target]

    if n_rows is not None:
        assert n_rows <= len(df_clean), (
            f"Requested {n_rows} rows but only {len(df_clean)} available."
        )
        combined = combined.sample(n=n_rows, random_state=42).reset_index(drop=True)
        df_clean = combined[df_clean.columns]
        y = combined[target].reset_index(drop=True)
        logger.info("Shuffled and sampled %d rows.", n_rows)

    constant_columns = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
    if constant_columns:
        df_clean = df_clean.drop(columns=constant_columns)
        logger.info("Removed constant columns: %s", constant_columns)

    le = LabelEncoder()
    columns_to_encode = df_clean.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    for column in columns_to_encode:
        df_clean[column] = le.fit_transform(df_clean[column])
        logger.info("Encoded column: %s", column)

    if y.dtype == "bool":
        y = y.astype(int)

    return df_clean, y


def hyper_param_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    task: str,
    cv: int = 3,
    n_iter: int = 5,
    n_jobs: int = -1,
    verbose: int = 0,
) -> dict:
    """Hyperparameter search using RandomizedSearchCV.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: 'RandomForest' or 'XGBoost'.
        task: 'classification' or 'regression'.
        cv: Number of cross-validation folds.
        n_iter: Number of parameter settings sampled.
        n_jobs: Number of parallel jobs (-1 for all cores).
        verbose: Verbosity level for RandomizedSearchCV.

    Returns:
        Best hyperparameters found.
    """
    estimator, param_distributions = _get_search_config(model_name, task)
    scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42,
    )
    search.fit(X, y)
    logger.info("%s %s best params: %s", model_name, task, search.best_params_)
    return search.best_params_
