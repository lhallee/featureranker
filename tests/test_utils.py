import numpy as np
import pandas as pd
import pytest

from featureranker.utils import (
    get_data,
    hyper_param_search,
    sanitize_column_names,
    view_data,
)


def test_sanitize_column_names():
    df = pd.DataFrame({"a b": [1], "c-d": [2], "e.f": [3], "ok": [4]})
    result = sanitize_column_names(df)
    assert list(result.columns) == ["a_b", "c_d", "e_f", "ok"]
    # original should not be mutated
    assert "a b" in df.columns


def test_sanitize_does_not_mutate_original():
    df = pd.DataFrame({"x y": [1]})
    result = sanitize_column_names(df)
    assert "x y" in df.columns
    assert "x_y" in result.columns


def test_view_data_no_nans(cancer_data):
    X, _ = cancer_data
    result = view_data(X)
    assert result.empty


def test_view_data_with_nans(messy_df):
    result = view_data(messy_df)
    assert "mostly_nan" in result.index
    assert result["mostly_nan"] == 90.0


def test_get_data_basic(messy_df):
    X, y = get_data(messy_df, target="target")
    assert "target" not in X.columns
    assert "constant" not in X.columns
    assert "mostly_nan" not in X.columns
    assert len(X) == len(y)
    # cat_col should be encoded as numeric
    assert X["cat_col"].dtype in (np.int32, np.int64)


def test_get_data_drop_columns(messy_df):
    X, y = get_data(messy_df, target="target", columns_to_drop=["num_c"])
    assert "num_c" not in X.columns


def test_get_data_n_rows(messy_df):
    X, y = get_data(messy_df, target="target", n_rows=20)
    assert len(X) == 20
    assert len(y) == 20


def test_get_data_missing_target(messy_df):
    with pytest.raises(AssertionError, match="not found"):
        get_data(messy_df, target="nonexistent")


def test_get_data_missing_columns(messy_df):
    with pytest.raises(AssertionError, match="not found"):
        get_data(messy_df, target="target", columns_to_drop=["fake_col"])


def test_hyper_param_search_rf_classification(cancer_data):
    X, y = cancer_data
    params = hyper_param_search(X, y, "RandomForest", "classification", cv=2, n_iter=2)
    assert isinstance(params, dict)
    assert "n_estimators" in params


def test_hyper_param_search_xgb_regression(diabetes_data):
    X, y = diabetes_data
    params = hyper_param_search(X, y, "XGBoost", "regression", cv=2, n_iter=2)
    assert isinstance(params, dict)
    assert "max_depth" in params


def test_hyper_param_search_invalid_model(cancer_data):
    X, y = cancer_data
    with pytest.raises(ValueError, match="Unknown model"):
        hyper_param_search(X, y, "InvalidModel", "classification")
