import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import f_regression, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor


def make_ranking(name, cols, importance):
    return pd.DataFrame({name:cols, 'Score':importance}).sort_values(by='Score', ascending=False).reset_index(drop=True)


def find_alpha_step(X, y, alpha_iter=100, init_step=1e-20):
    coef_count, total_iter, step_a = X.shape[1], 0, init_step
    pbar = tqdm(total=30, desc='Finding Alpha Step')
    while coef_count > 0 and total_iter < 1000000:
        alpha = step_a
        lasso = Lasso(alpha=alpha*alpha_iter, max_iter=1000)
        lasso.fit(X, y)
        alpha += step_a
        coef_count = np.count_nonzero(lasso.coef_)
        if coef_count > 0:
            step_a = step_a * 10
            total_iter += 1
            pbar.update(1)
    return step_a / 1000


def find_c_step(X, y, init_step=1e20):
    coef_count, total_iter, step_c = X.shape[1], 0, init_step
    pbar = tqdm(total=30, desc='Finding C Step')
    while coef_count > 0 and total_iter < 1000000:
        c = step_c
        log = LogisticRegression(penalty='l1', C=c, solver='liblinear')
        log.fit(X, y)
        coef_count = np.count_nonzero(log.coef_)
        if coef_count > 0:
            step_c = step_c / 10
            total_iter += 1
            pbar.update(1)
    return step_c / 10, step_c * 1000


def l1_classification_ranking(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    step_c, C = find_c_step(X_scaled, y)
    last_coef = np.ones(X.shape[1])  # Initialize last_coef to all ones
    # Initialize DataFrame to store feature ranking
    ranking = pd.DataFrame(columns=['L1', 'Score'])
    # Initialize set to store features that have been dropped
    dropped_features = set()
    # Scale the features using Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Loop until all feature coefficients become zero
    pbar = tqdm(total=10000, desc='Ranking l1 Classification') # 10000 because that is initial C / step_c
    while np.any(last_coef != 0) and C > 0:
        # Initialize and fit the Logistic Regression model with L1 penalty
        log_reg = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42)
        log_reg.fit(X_scaled, y)
        coef = log_reg.coef_.flatten()
        # Identify features whose coefficients have just become zero
        just_zeroed = (last_coef != 0) & (coef == 0)
        zeroed_features = X.columns[just_zeroed].tolist()
        # Update the feature ranking
        new_rankings = pd.DataFrame({'L1': zeroed_features, 'Score': [C] * len(zeroed_features)})
        for feature in zeroed_features:
            if feature not in dropped_features:
                ranking = pd.concat([ranking, new_rankings[new_rankings['L1'] == feature]], ignore_index=True)
                dropped_features.add(feature)
        last_coef = coef
        C -= step_c
        pbar.update(1)
    # Sort features by importance (descending order of C when they were dropped)
    ranking = ranking.sort_values(by='Score', ascending=True).reset_index(drop=True)
    return ranking


def l1_regression_ranking(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    step_a = find_alpha_step(X_scaled, y)
    last_coef = np.ones(X.shape[1])
    ranking = pd.DataFrame(columns=['Feature', 'Score'])
    dropped_features = set()
    alpha = step_a
    max_iter = 0
    pbar = tqdm(total=10000, desc='Ranking l1 Regression')
    while np.any(last_coef != 0) and max_iter < 1000000:
        las = Lasso(alpha=alpha, max_iter=10000)
        las.fit(X_scaled, y)
        coef = las.coef_

        # Identify features whose coefficients have just become zero
        just_zeroed = (last_coef != 0) & (coef == 0)
        zeroed_features = X.columns[just_zeroed].tolist()

        # Update the feature ranking
        new_rankings = pd.DataFrame({'Feature': zeroed_features, 'Score': [alpha] * len(zeroed_features)})
        for feature in zeroed_features:
            if feature not in dropped_features:
                ranking = pd.concat([ranking, new_rankings[new_rankings['Feature'] == feature]], ignore_index=True)
                dropped_features.add(feature)

        # Update last_coef to current coef for the next iteration
        last_coef = coef

        # Increment the alpha value
        alpha += step_a
        max_iter += 1
        pbar.update(1)

    # Sort features by importance (ascending order of when they were dropped)
    ranking = ranking.sort_values(by='Score', ascending=False).reset_index(drop=True)

    return ranking


def classification_ranking(X, y, rf_hyper, xb_hyper):
    cols = X.columns
    rf = make_ranking('RF', cols, RandomForestClassifier(**rf_hyper).fit(X, y).feature_importances_)
    xg = make_ranking('XG', cols, XGBClassifier(use_label_encoder=False, **xb_hyper).fit(X, y).feature_importances_)
    mi = make_ranking('MI', cols, mutual_info_classif(X, y))
    f = make_ranking('F', cols, np.nan_to_num(f_classif(X, y)[0]))
    l1 = l1_classification_ranking(X, y)
    return pd.concat([rf, xg, mi, f, l1], axis=1).reset_index(drop=True)


def regression_ranking(X, y, rf_hyper, xb_hyper):
    cols = X.columns
    rf = make_ranking('RF', cols, RandomForestRegressor(**rf_hyper).fit(X, y).feature_importances_)
    xg = make_ranking('XG', cols, XGBRegressor(use_label_encoder=False, **xb_hyper).fit(X, y).feature_importances_)
    mi = make_ranking('MI', cols, mutual_info_regression(X, y))
    f = make_ranking('F', cols, np.nan_to_num(f_regression(X, y)[0]))
    l1 = l1_regression_ranking(X, y)
    return pd.concat([rf, xg, mi, f, l1], axis=1).reset_index(drop=True)