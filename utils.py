import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer

from plots import *


model_params = {
    'classification': {
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'max_depth': [int(x) for x in np.linspace(3, 50, num=10)],
                'min_child_weight': [int(x) for x in np.linspace(1, 10, num=10)],
                'gamma': [float(x) for x in np.linspace(0, 0.5, num=10)],
                'subsample': [float(x) for x in np.linspace(0.5, 1.0, num=10)],
                'colsample_bytree': [float(x) for x in np.linspace(0.5, 1.0, num=10)],
                'learning_rate': [float(x) for x in np.logspace(np.log10(0.01), np.log10(0.5), base=10, num=10)],
                'n_estimators': [int(x) for x in np.logspace(np.log10(100), np.log10(1000), base=10, num=10)],
                'reg_alpha': [float(x) for x in np.logspace(np.log10(0.1), np.log10(100), base=10, num=10)],
                'reg_lambda': [float(x) for x in np.logspace(np.log10(0.1), np.log10(100), base=10, num=10)],
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [int(x) for x in np.linspace(10, 1000, num=10)],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [int(x) for x in np.linspace(10, 100, num=10)] + [None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
            }
        }
    },
    'regression': {
        'XGBoost': {
            'model': XGBRegressor(),
            'params': {
                'max_depth': [int(x) for x in np.linspace(3, 50, num=10)],
                'min_child_weight': [int(x) for x in np.linspace(1, 10, num=10)],
                'gamma': [float(x) for x in np.linspace(0, 0.5, num=10)],
                'subsample': [float(x) for x in np.linspace(0.5, 1.0, num=10)],
                'colsample_bytree': [float(x) for x in np.linspace(0.5, 1.0, num=10)],
                'learning_rate': [float(x) for x in np.logspace(np.log10(0.01), np.log10(0.5), base=10, num=10)],
                'n_estimators': [int(x) for x in np.logspace(np.log10(100), np.log10(1000), base=10, num=10)],
                'reg_alpha': [float(x) for x in np.logspace(np.log10(0.1), np.log10(100), base=10, num=10)],
                'reg_lambda': [float(x) for x in np.logspace(np.log10(0.1), np.log10(100), base=10, num=10)],
            }
        },
        'Random_Forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [int(x) for x in np.linspace(10, 1000, num=10)],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [int(x) for x in np.linspace(10, 100, num=10)] + [None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
            }
        }
    }
}


def sanitize_column_names(df): # remove typical unwanted characters from column names
    df.columns = [col.translate(str.maketrans('[]<>{}', '____')) for col in df.columns]
    return df


def view_data(df):
    for column in df.columns:
        nan_count = df[column].isna().sum()
        nan_percentage = round(nan_count / len(df) * 100, 1)
        print(f'The column {column} has {nan_percentage}% NaN values.')


def get_data(df, labels, thresh=0.8, columns_to_drop=None):
    y = df[labels]
    df_clean = df.drop(columns=columns_to_drop + labels if columns_to_drop is not None else labels)
    threshold = thresh * len(df_clean)
    df_clean = df_clean.dropna(axis=1, thresh=threshold)
    combined = pd.concat([df_clean, y], axis=1)
    combined_clean = combined.dropna()
    df_clean = combined_clean[df_clean.columns]
    y = combined_clean[labels]
    le = LabelEncoder()
    columns_to_encode = df_clean.select_dtypes(include=['object', 'string']).columns
    for column in columns_to_encode:
        df_clean[column] = le.fit_transform(df_clean[column])
    X = df_clean
    return X, y


def spearman_scoring_function(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def regression_hyper_param_search(X, y, cv, num_runs, model_params=model_params):
    spearman_scorer = make_scorer(spearman_scoring_function, greater_is_better=True)
    total_hypers = []
    for model_name, mp in model_params['regression'].items():
        # Initialize the RandomizedSearchCV object
        clf = RandomizedSearchCV(mp['model'],
                                 mp['params'],
                                 n_iter=num_runs,
                                 cv=cv,
                                 scoring=spearman_scorer,
                                 random_state=42,
                                 verbose=2,
                                 n_jobs=-1)
        clf.fit(X, y)
        print(f'Best spearmn for {model_name}: {clf.best_score_}')
        total_hypers.append({'model': model_name, 'best_score': clf.best_score_, 'best_params': clf.best_params_})
    return total_hypers


def classification_hyper_param_search(X, y, cv, num_runs, model_params=model_params):
    total_hypers = []
    for model_name, mp in model_params['classification'].items():
        # Initialize the RandomizedSearchCV object
        clf = RandomizedSearchCV(mp['model'],
                                 mp['params'],
                                 n_iter=num_runs,
                                 cv=cv,
                                 random_state=42,
                                 verbose=2,
                                 n_jobs=-1)
        clf.fit(X, y)
        predictions = cross_val_predict(clf.best_estimator_, X, y, cv=cv, n_jobs=-1)
        acc = accuracy_score(y, predictions)
        cm = confusion_matrix(y, predictions)
        plot_confusion_matrix(cm, title=f'{model_name}_{round(acc, 3)}', labels=np.unique(y))
    total_hypers.append({'model': model_name, 'best_score': clf.best_score_, 'best_params': clf.best_params_})
    return total_hypers




