import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.feature_selection import f_regression, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c
from xgboost import XGBClassifier, XGBRegressor
from tqdm.auto import tqdm

from featureranker.utils import classification_hyper_param_search, regression_hyper_param_search


def make_ranking(name, cols, importance):
    return pd.DataFrame({name:cols, 'Score':importance}).sort_values(by='Score', ascending=False).reset_index(drop=True)


def l1_classification_ranking(X, y, scale=10, num_alphas=100, max_iter=100, norm=True):
    counts, avg = {}, {}
    columns = X.columns.tolist()
    if norm:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    c = l1_min_c(X, y, loss='log')
    cs = c * np.logspace(0, scale, num_alphas)
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        tol=1e-5,
        max_iter=max_iter,
        intercept_scaling=1e5,
    )

    coefs_ = []
    for c in tqdm(cs, desc='L1 ranking'):
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())
    coefs = np.array(coefs_)

    for coef, col in zip(coefs.T, columns): # needs transpose for logistic coefs
        counts[col] = np.count_nonzero(coef)
        avg[col] = np.abs(np.mean(coef))
    
    ranking = pd.DataFrame(list(counts.items()), columns=['L1', 'Score'])
    ranking['Avg'] = ranking['L1'].map(avg) # average coefficient is tiebreaker
    ranking = ranking.sort_values(by=['Score', 'Avg'], ascending=[False, False])
    return ranking[['L1', 'Score']]


def l1_regression_ranking(X, y, scale=1e-20, num_alphas=1000, norm=True, verbose=False):
    counts, avg = {}, {}
    columns = X.columns.tolist()
    if norm:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    coefs = lasso_path(X, y, eps=scale, n_alphas=num_alphas, verbose=verbose)[1]
    
    for coef, col in zip(coefs, columns):
        counts[col] = np.count_nonzero(coef)
        avg[col] = np.abs(np.mean(coef))

    ranking = pd.DataFrame(list(counts.items()), columns=['L1', 'Score'])
    ranking['Avg'] = ranking['L1'].map(avg) # average coefficient is tiebreaker
    ranking = ranking.sort_values(by=['Score', 'Avg'], ascending=[False, False])
    return ranking[['L1', 'Score']]


def classification_ranking(X, y,
                           cv=3,
                           num_runs=5,
                           save=False,
                           predict=True,
                           norm=True,
                           scale=10,
                           num_alphas=100,
                           max_iter=100,
                           choices=['rf', 'xg', 'mi', 'f_test', 'l1']):
    cols = X.columns
    rankings = []
    if 'rf' in choices:
        rf_hyper = classification_hyper_param_search(X, y, 'RandomForest', cv, num_runs, save=save, predict=predict)
        rf = make_ranking('RF', cols, RandomForestClassifier(**rf_hyper).fit(X, y).feature_importances_)
        rankings.append(('RF', rf))
    if 'xg' in choices:
        xg_hyper = classification_hyper_param_search(X, y, 'XGBoost', cv, num_runs, save=save, predict=predict)
        xg = make_ranking('XG', cols, XGBClassifier(use_label_encoder=False, **xg_hyper).fit(X, y).feature_importances_)
        rankings.append(('XG', xg))
    if 'mi' in choices:
        mi = make_ranking('MI', cols, mutual_info_classif(X, y))
        rankings.append(('MI', mi))
    if 'f_test' in choices:
        f = make_ranking('F', cols, np.nan_to_num(f_classif(X, y)[0]))
        rankings.append(('F', f))
    if 'l1' in  choices:
        l1 = l1_classification_ranking(X, y, norm=norm, scale=scale, num_alphas=num_alphas, max_iter=max_iter)
        rankings.append(('L1', l1))
    if not rankings:
        print('You did not pass any correct choices. Try again.')
    return rankings


def regression_ranking(X, y,
                       cv=3,
                       num_runs=5,
                       save=False,
                       predict=True,
                       norm=True,
                       verbose=False,
                       scale=1e-20,
                       num_alphas=100,
                       choices=['rf', 'xg', 'mi', 'f_test', 'l1']):
    cols = X.columns
    rankings = []
    if 'rf' in choices:
        rf_hyper = regression_hyper_param_search(X, y, 'RandomForest', cv, num_runs, save=save, predict=predict)
        rf = make_ranking('RF', cols, RandomForestRegressor(**rf_hyper).fit(X, y).feature_importances_)
        rankings.append(('RF', rf))
    if 'xg' in choices:
        xg_hyper = regression_hyper_param_search(X, y, 'XGBoost', cv, num_runs, save=save, predict=predict)
        xg = make_ranking('XG', cols, XGBRegressor(use_label_encoder=False, **xg_hyper).fit(X, y).feature_importances_)
        rankings.append(('XG', xg))
    if 'mi' in choices:
        mi = make_ranking('MI', cols, mutual_info_regression(X, y))
        rankings.append(('MI', mi))
    if 'f_test' in choices:
        f = make_ranking('F', cols, np.nan_to_num(f_regression(X, y)[0]))
        rankings.append(('F', f))
    if 'l1' in choices:
        l1 = l1_regression_ranking(X, y, norm=norm, verbose=verbose)
        rankings.append(('L1', l1))
    if not rankings:
        print('You did not pass any correct choices. Try again.')
    return rankings


def voting(rankings, weights=(0.2, 0.2, 0.2, 0.2, 0.2)):
    if len(set(weights)) > 1:
        assert len(rankings) == len(weights), 'Specify the same number of weights as ranking methods you used.\nSee documentation for details.'
    else:
        weights = weights[:len(rankings)]
    final_scores = {}
    for (ranking_name, ranking), weight in zip(rankings, weights):
        feature_list = ranking[ranking_name].tolist()
        for i, feature in enumerate(reversed(feature_list)):
            if feature not in final_scores:
                final_scores[feature] = 0
            final_scores[feature] += (i + 1) * weight
    final_ranking = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    return final_ranking