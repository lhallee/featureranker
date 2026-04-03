import logging
import math
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c
from xgboost import XGBClassifier, XGBRegressor

from .utils import hyper_param_search

logger = logging.getLogger(__name__)


def make_ranking(name: str, cols: list[str], importance: np.ndarray) -> pd.DataFrame:
    """Create a DataFrame ranking features by importance scores."""
    assert len(cols) == len(importance), (
        f"Length mismatch: {len(cols)} columns vs {len(importance)} scores."
    )
    return (
        pd.DataFrame({name: cols, "Score": importance})
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )


def l1_regression_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank features via L1-regularized linear regression (lasso path)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas, coefs, _ = lasso_path(X_scaled, y)
    alphas = alphas[::-1]
    coefs = coefs[:, ::-1]

    zeroing_alphas = []
    for i in range(coefs.shape[0]):
        non_zero = np.where(coefs[i, :] != 0)[0]
        if non_zero.size > 0:
            zeroing_alphas.append(alphas[non_zero[-1]])
        else:
            zeroing_alphas.append(alphas[0])

    return (
        pd.DataFrame({"L1": X.columns, "Score": zeroing_alphas})
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )


def l1_classification_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    n_regularization_steps: int = 50,
) -> pd.DataFrame:
    """Rank features via L1-regularized logistic regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    min_c = l1_min_c(X_scaled, y, loss="log")
    Cs = min_c * np.logspace(0, 3, n_regularization_steps)

    clf = LogisticRegression(
        l1_ratio=1.0,
        solver="saga",
        tol=1e-6,
        max_iter=int(1e6),
        warm_start=True,
        intercept_scaling=1.0,
    )

    coefs_ = []
    for C in Cs:
        clf.set_params(C=C)
        clf.fit(X_scaled, y)
        coefs_.append(clf.coef_.ravel().copy())

    coefs_ = np.array(coefs_)

    zeroing_Cs = []
    for i in range(coefs_.shape[1]):
        non_zero = np.where(coefs_[:, i] != 0)[0]
        if non_zero.size > 0:
            zeroing_Cs.append(Cs[non_zero[-1]])
        else:
            zeroing_Cs.append(Cs[0])

    return (
        pd.DataFrame({"L1": X.columns, "Score": zeroing_Cs})
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )


# --- Private ranking functions for parallel dispatch ---


def _rank_rf(
    X: pd.DataFrame, y: pd.Series, task: str, **kwargs
) -> tuple[str, pd.DataFrame]:
    logger.info("Running Random Forest %s ranking...", task)
    params = hyper_param_search(
        X, y, "RandomForest", task,
        cv=kwargs.get("cv", 3),
        n_iter=kwargs.get("n_iter", 5),
        n_jobs=kwargs.get("search_n_jobs", -1),
        verbose=kwargs.get("verbose", 0),
    )
    if task == "classification":
        model = RandomForestClassifier(random_state=42, **params)
    else:
        model = RandomForestRegressor(random_state=42, **params)
    model.fit(X, y)
    ranking = make_ranking("RF", X.columns.tolist(), model.feature_importances_)
    logger.info("Completed Random Forest %s ranking.", task)
    return ("RF", ranking)


def _rank_xg(
    X: pd.DataFrame, y: pd.Series, task: str, **kwargs
) -> tuple[str, pd.DataFrame]:
    logger.info("Running XGBoost %s ranking...", task)
    params = hyper_param_search(
        X, y, "XGBoost", task,
        cv=kwargs.get("cv", 3),
        n_iter=kwargs.get("n_iter", 5),
        n_jobs=kwargs.get("search_n_jobs", -1),
        verbose=kwargs.get("verbose", 0),
    )
    if task == "classification":
        model = XGBClassifier(eval_metric="logloss", random_state=42, **params)
    else:
        model = XGBRegressor(random_state=42, **params)
    model.fit(X, y)
    ranking = make_ranking("XG", X.columns.tolist(), model.feature_importances_)
    logger.info("Completed XGBoost %s ranking.", task)
    return ("XG", ranking)


def _rank_mi(
    X: pd.DataFrame, y: pd.Series, task: str, **kwargs
) -> tuple[str, pd.DataFrame]:
    logger.info("Running Mutual Information %s ranking...", task)
    if task == "classification":
        scores = mutual_info_classif(X, y, random_state=42)
    else:
        scores = mutual_info_regression(X, y, random_state=42)
    ranking = make_ranking("MI", X.columns.tolist(), scores)
    logger.info("Completed Mutual Information %s ranking.", task)
    return ("MI", ranking)


def _rank_f(
    X: pd.DataFrame, y: pd.Series, task: str, **kwargs
) -> tuple[str, pd.DataFrame]:
    logger.info("Running F-test %s ranking...", task)
    if task == "classification":
        scores, _ = f_classif(X, y)
    else:
        scores, _ = f_regression(X, y)
    scores = np.nan_to_num(scores)
    ranking = make_ranking("F", X.columns.tolist(), scores)
    logger.info("Completed F-test %s ranking.", task)
    return ("F", ranking)


def _rank_l1(
    X: pd.DataFrame, y: pd.Series, task: str, **kwargs
) -> tuple[str, pd.DataFrame]:
    logger.info("Running L1 %s ranking...", task)
    n_steps = kwargs.get("n_regularization_steps", 50)
    if task == "classification":
        ranking = l1_classification_ranking(X, y, n_regularization_steps=n_steps)
    else:
        ranking = l1_regression_ranking(X, y)
    logger.info("Completed L1 %s ranking.", task)
    return ("L1", ranking)


_RANKER_DISPATCH = {
    "rf": _rank_rf,
    "xg": _rank_xg,
    "mi": _rank_mi,
    "f_test": _rank_f,
    "l1": _rank_l1,
}

VALID_CHOICES = frozenset(_RANKER_DISPATCH.keys())


def feature_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    choices: list[str] | None = None,
    n_jobs: int = 1,
    save: bool = False,
    save_path: str | None = None,
    **kwargs,
) -> list[tuple[str, pd.DataFrame]]:
    """Run an ensemble of feature ranking methods.

    Args:
        X: Feature matrix.
        y: Target vector.
        task: 'classification' or 'regression'.
        choices: Ranking methods to use. Default: all five.
            Options: 'rf', 'xg', 'mi', 'f_test', 'l1'.
        n_jobs: Parallel jobs for dispatching rankers. 1=sequential, -1=all cores.
        save: Save rankings to a pickle file.
        save_path: Path for the pickle file.
        **kwargs: Passed to individual rankers (cv, n_iter, verbose, n_regularization_steps, search_n_jobs).

    Returns:
        List of (method_name, ranking_dataframe) tuples.
    """
    assert task in ("classification", "regression"), (
        f"Invalid task: {task}. Use 'classification' or 'regression'."
    )

    if choices is None:
        choices = ["rf", "xg", "mi", "f_test", "l1"]

    invalid = set(choices) - VALID_CHOICES
    assert not invalid, f"Invalid choices: {invalid}. Valid: {VALID_CHOICES}"

    logger.info("Feature ranking: task=%s, methods=%s, n_jobs=%d", task, choices, n_jobs)

    funcs = [_RANKER_DISPATCH[c] for c in choices]
    rankings = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fn)(X, y, task, **kwargs) for fn in funcs
    )

    logger.info("Feature ranking completed.")

    if save:
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"feature_rankings_{task}_{timestamp}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(rankings, f)
        logger.info("Rankings saved to %s", save_path)

    return rankings


def voting(
    rankings: list[tuple[str, pd.DataFrame]],
    weights: list[float] | None = None,
    method: str = "reciprocal_rank",
    save: bool = False,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Aggregate feature rankings using a weighted voting scheme.

    Args:
        rankings: List of (method_name, ranking_df) tuples from feature_ranking().
        weights: Weight per ranking method. Default: equal weights (1.0 each).
        method: Voting method. Options:
            'reciprocal_rank' (default): weight * (1 / rank)
            'borda': weight * (n_features - rank)
            'exponential': weight * exp(-rank / n_features)
        save: Save result to CSV.
        save_path: Path for CSV file.

    Returns:
        DataFrame with 'Feature' and 'Score' columns, sorted by score descending.
    """
    assert rankings, "Rankings list is empty."
    valid_methods = {"reciprocal_rank", "borda", "exponential"}
    assert method in valid_methods, f"Invalid method: {method}. Use one of {valid_methods}"

    if weights is None:
        weights = [1.0] * len(rankings)
    assert len(weights) == len(rankings), (
        f"Length mismatch: {len(weights)} weights vs {len(rankings)} rankings."
    )

    score_dict: dict[str, float] = defaultdict(float)

    for (method_name, ranking_df), weight in zip(rankings, weights):
        feature_list = ranking_df[method_name].tolist()
        n_features = len(feature_list)
        for rank, feature in enumerate(feature_list, start=1):
            if method == "reciprocal_rank":
                score = weight * (1.0 / rank)
            elif method == "borda":
                score = weight * (n_features - rank)
            else:  # exponential
                score = weight * math.exp(-rank / n_features)
            score_dict[feature] += score

    final_ranking = (
        pd.DataFrame.from_dict(score_dict, orient="index", columns=["Score"])
        .sort_values(by="Score", ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    if save:
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"aggregated_ranking_{timestamp}.csv"
        final_ranking.to_csv(save_path, index=False)
        logger.info("Aggregated ranking saved to %s", save_path)

    return final_ranking
