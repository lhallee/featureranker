import pandas as pd
import numpy as np
import warnings
import pickle
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.feature_selection import (
    f_regression,
    f_classif,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c
from xgboost import XGBClassifier, XGBRegressor

from featureranker.utils import (
    classification_hyper_param_search,
    regression_hyper_param_search,
)


def make_ranking(name: str, cols: List[str], importance: np.ndarray) -> pd.DataFrame:
    """
    Create a DataFrame ranking features based on their importance scores.

    Parameters:
        name (str): The name of the ranking method.
        cols (List[str]): List of feature names.
        importance (np.ndarray): Importance scores corresponding to the features.

    Returns:
        pd.DataFrame: DataFrame sorted by the importance scores in descending order.
    """
    if len(cols) != len(importance):
        raise ValueError("The length of 'cols' and 'importance' must be the same.")
    return (
        pd.DataFrame({name: cols, "Score": importance})
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )


def l1_regression_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Perform feature ranking using L1-regularized linear regression by increasing penalization
    until each coefficient is zeroed out.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        pd.DataFrame: DataFrame containing features and the penalization level at which
                      they are zeroed out.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute the Lasso path
    alphas, coefs, _ = lasso_path(X_scaled, y)

    # Reverse alphas and coefs to have increasing penalization
    alphas = alphas[::-1]
    coefs = coefs[:, ::-1]

    # For each feature, find the maximum alpha where the coefficient is non-zero
    zeroing_alphas = []
    for i in range(coefs.shape[0]):
        coef_i = coefs[i, :]
        non_zero_indices = np.where(coef_i != 0)[0]
        if non_zero_indices.size > 0:
            zero_alpha = alphas[non_zero_indices[-1]]  # Last alpha where coef is non-zero
        else:
            zero_alpha = alphas[0]
        zeroing_alphas.append(zero_alpha)

    ranking = pd.DataFrame({'L1': X.columns, 'Score': zeroing_alphas})
    ranking.sort_values(by='Score', ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    return ranking


def l1_classification_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Perform feature ranking using L1-regularized logistic regression by increasing penalization
    until each coefficient is zeroed out.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        pd.DataFrame: DataFrame containing features and the penalization level at which
                      they are zeroed out.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute the minimal value of C that will zero out all coefficients
    min_c = l1_min_c(X_scaled, y, loss='log')

    # Define a range of Cs (inverse of regularization strength)
    Cs = min_c * np.logspace(0, 3, 100)  # Adjust the range as necessary

    # Initialize logistic regression model
    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        tol=1e-6,
        max_iter=int(1e6),
        warm_start=True,
        intercept_scaling=1.0,
    )

    # Store coefficients for each C
    coefs_ = []

    for C in Cs:
        clf.set_params(C=C)
        clf.fit(X_scaled, y)
        coefs_.append(clf.coef_.ravel().copy())

    coefs_ = np.array(coefs_)  # Shape: (n_Cs, n_features)

    # For each feature, find the maximum C where the coefficient is non-zero
    zeroing_Cs = []
    for i in range(coefs_.shape[1]):
        coef_i = coefs_[:, i]
        non_zero_indices = np.where(coef_i != 0)[0]
        if non_zero_indices.size > 0:
            zero_C = Cs[non_zero_indices[-1]]  # Last C where coef is non-zero
        else:
            zero_C = Cs[0]
        zeroing_Cs.append(zero_C)

    ranking = pd.DataFrame({'L1': X.columns, 'Score': zeroing_Cs})
    ranking.sort_values(by='Score', ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    return ranking


def feature_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    choices: Optional[List[str]] = None,
    save: bool = False,
    save_path: Optional[str] = None,
    **kwargs,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    General feature ranking function for both classification and regression tasks.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        task (str): 'classification' or 'regression'.
        choices (List[str]): List of ranking methods to use.
        save (bool): Whether to save the rankings to a pickle file.
        save_path (str): Path to save the pickle file. If None, a default name is used.
        **kwargs: Additional keyword arguments for hyperparameter search functions.

    Returns:
        List[Tuple[str, pd.DataFrame]]: List of tuples containing ranking method names and their rankings.
    """
    print(f"Starting feature ranking for task: {task.capitalize()}")

    if choices is None:
        choices = ["rf", "xg", "mi", "f_test", "l1"]
        print("No choices provided. Using default ranking methods:", choices)
    else:
        print("Ranking methods selected:", choices)

    valid_choices = {"rf", "xg", "mi", "f_test", "l1"}
    invalid_choices = set(choices) - valid_choices
    if invalid_choices:
        raise ValueError(f"Invalid choices provided: {invalid_choices}")

    cols = X.columns
    rankings = []

    if task == "classification":
        if "rf" in choices:
            print("Starting Random Forest classification ranking...")
            rf_hyper = classification_hyper_param_search(X, y, "RandomForest", **kwargs)
            model = RandomForestClassifier(**rf_hyper)
            model.fit(X, y)
            rf = make_ranking("RF", cols.tolist(), model.feature_importances_)
            rankings.append(("RF", rf))
            print("Completed Random Forest classification ranking.")
        
        if "xg" in choices:
            print("Starting XGBoost classification ranking...")
            xg_hyper = classification_hyper_param_search(X, y, "XGBoost", **kwargs)
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xg_hyper)
            model.fit(X, y)
            xg = make_ranking("XG", cols.tolist(), model.feature_importances_)
            rankings.append(("XG", xg))
            print("Completed XGBoost classification ranking.")
        
        if "mi" in choices:
            print("Starting Mutual Information classification ranking...")
            mi_scores = mutual_info_classif(X, y)
            mi = make_ranking("MI", cols.tolist(), mi_scores)
            rankings.append(("MI", mi))
            print("Completed Mutual Information classification ranking.")
        
        if "f_test" in choices:
            print("Starting F-Test classification ranking...")
            f_scores, _ = f_classif(X, y)
            f_scores = np.nan_to_num(f_scores)
            f = make_ranking("F", cols.tolist(), f_scores)
            rankings.append(("F", f))
            print("Completed F-Test classification ranking.")
        
        if "l1" in choices:
            print("Starting L1 classification ranking...")
            l1 = l1_classification_ranking(X, y, **kwargs)
            rankings.append(("L1", l1))
            print("Completed L1 classification ranking.")
    
    elif task == "regression":
        if "rf" in choices:
            print("Starting Random Forest regression ranking...")
            rf_hyper = regression_hyper_param_search(X, y, "RandomForest", **kwargs)
            model = RandomForestRegressor(**rf_hyper)
            model.fit(X, y)
            rf = make_ranking("RF", cols.tolist(), model.feature_importances_)
            rankings.append(("RF", rf))
            print("Completed Random Forest regression ranking.")
        
        if "xg" in choices:
            print("Starting XGBoost regression ranking...")
            xg_hyper = regression_hyper_param_search(X, y, "XGBoost", **kwargs)
            model = XGBRegressor(**xg_hyper)
            model.fit(X, y)
            xg = make_ranking("XG", cols.tolist(), model.feature_importances_)
            rankings.append(("XG", xg))
            print("Completed XGBoost regression ranking.")
        
        if "mi" in choices:
            print("Starting Mutual Information regression ranking...")
            mi_scores = mutual_info_regression(X, y)
            mi = make_ranking("MI", cols.tolist(), mi_scores)
            rankings.append(("MI", mi))
            print("Completed Mutual Information regression ranking.")
        
        if "f_test" in choices:
            print("Starting F-Test regression ranking...")
            f_scores, _ = f_regression(X, y)
            f_scores = np.nan_to_num(f_scores)
            f = make_ranking("F", cols.tolist(), f_scores)
            rankings.append(("F", f))
            print("Completed F-Test regression ranking.")
        
        if "l1" in choices:
            print("Starting L1 regression ranking...")
            l1 = l1_regression_ranking(X, y, **kwargs)
            rankings.append(("L1", l1))
            print("Completed L1 regression ranking.")
    
    else:
        raise ValueError("Invalid task specified. Choose 'classification' or 'regression'.")

    if not rankings:
        warnings.warn("No valid choices were provided for ranking methods.")
    else:
        print("Feature ranking completed successfully.")

    if save:
        if save_path is None:
            # Create a default filename based on task and current timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"feature_rankings_{task}_{timestamp}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(rankings, f)
        print(f"Rankings saved to {save_path}")

    return rankings


def voting(
    rankings: List[Tuple[str, pd.DataFrame]],
    weights: Optional[List[float]] = None,
    save: bool = False,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate feature rankings using a weighted voting scheme.

    Parameters:
        rankings (List[Tuple[str, pd.DataFrame]]): List of tuples containing method names and their rankings.
        weights (List[float]): List of weights corresponding to each ranking method.
        save (bool): Whether to save the aggregated ranking to a pickle file.
        save_path (str): Path to save the pickle file. If None, a default name is used.

    Returns:
        pd.DataFrame: DataFrame containing features and their aggregated scores.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    else:
        if len(weights) != len(rankings):
            raise ValueError("Length of weights must match the number of rankings.")

    score_dict = {}
    for (method_name, ranking_df), weight in zip(rankings, weights):
        feature_list = ranking_df[method_name].tolist()
        for rank, feature in enumerate(feature_list, start=1):
            score = weight * (1 / rank)  # Higher rank contributes more
            score_dict[feature] = score_dict.get(feature, 0) + score

    final_ranking = (
        pd.DataFrame.from_dict(score_dict, orient="index", columns=["Score"])
        .sort_values(by="Score", ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    if save:
        if save_path is None:
            # Create a default filename based on current timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"aggregated_ranking_{timestamp}.csv"
        final_ranking.to_csv(save_path, index=False)
        print(f"Aggregated ranking saved to {save_path}")

    return final_ranking
