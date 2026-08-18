"""Tests for dataset preparation."""

import numpy as np
import pandas as pd
import pytest

from featureranker.data import get_data, sanitize_column_names, view_data


def test_sanitize_column_names():
    df = pd.DataFrame({"a b": [1], "c-d": [2], "e.f": [3], "ok": [4]})
    result = sanitize_column_names(df)
    assert list(result.columns) == ["a_b", "c_d", "e_f", "ok"]


def test_sanitize_does_not_mutate_original():
    df = pd.DataFrame({"x y": [1]})
    result = sanitize_column_names(df)
    assert "x y" in df.columns
    assert "x_y" in result.columns


def test_view_data_no_nans(cancer_data):
    X, _ = cancer_data
    assert view_data(X).empty


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


def test_get_data_onehot_default(messy_df):
    """cat_col holds A/B/C strings, so it expands into named sub-features."""
    X, _ = get_data(messy_df, target="target")
    assert "cat_col" not in X.columns
    sub_features = X[["cat_col-A", "cat_col-B", "cat_col-C"]]
    assert (sub_features.sum(axis=1) == 1).all()
    assert set(sub_features.to_numpy().ravel()) == {0, 1}


def test_get_data_onehot_injected_in_place(messy_df):
    """Sub-features sit where the parent column was, not appended at the end."""
    X, _ = get_data(messy_df, target="target")
    assert list(X.columns) == [
        "num_a", "num_b", "num_c", "cat_col-A", "cat_col-B", "cat_col-C",
    ]


def test_get_data_onehot_int_categories():
    """Integer categories name sub-features color-0, color-1, ..."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "color": pd.Categorical([0, 1, 2, 0]),
        "target": [0, 1, 0, 1],
    })
    X, _ = get_data(df, target="target")
    assert list(X.columns) == ["num", "color-0", "color-1", "color-2"]


def test_get_data_label_encoding_option(messy_df):
    X, _ = get_data(messy_df, target="target", encoding="label")
    assert "cat_col" in X.columns
    assert np.issubdtype(X["cat_col"].dtype, np.integer)


def test_get_data_max_categories_fallback():
    """Cardinality above max_categories falls back to label encoding."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        "wide": ["a", "b", "c", "d", "e"],
        "target": [0, 1, 0, 1, 0],
    })
    X, _ = get_data(df, target="target", max_categories=3)
    assert "wide" in X.columns
    assert np.issubdtype(X["wide"].dtype, np.integer)
    X_full, _ = get_data(df, target="target", max_categories=None)
    assert "wide-a" in X_full.columns
    assert X_full.shape[1] == 6


def test_get_data_bool_column_stays_single():
    """Booleans become one 0/1 column, not two redundant sub-features."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "flag": [True, False, True, False],
        "target": [0, 1, 0, 1],
    })
    X, _ = get_data(df, target="target")
    assert list(X.columns) == ["num", "flag"]
    assert set(X["flag"]) == {0, 1}


def test_get_data_bad_encoding(messy_df):
    with pytest.raises(ValueError, match="encoding"):
        get_data(messy_df, target="target", encoding="ordinal")


def test_get_data_bad_max_categories(messy_df):
    with pytest.raises(ValueError, match="max_categories"):
        get_data(messy_df, target="target", max_categories=1)


def test_get_data_duplicate_subfeature_name():
    """A sub-feature name colliding with a real column raises, not silently duplicates."""
    df = pd.DataFrame({
        "color-blue": [1.0, 2.0, 3.0, 4.0],
        "color": ["blue", "red", "blue", "red"],
        "target": [0, 1, 0, 1],
    })
    with pytest.raises(ValueError, match="duplicate"):
        get_data(df, target="target")


def test_get_data_drop_columns(messy_df):
    X, _ = get_data(messy_df, target="target", columns_to_drop=["num_c"])
    assert "num_c" not in X.columns


def test_get_data_n_rows(messy_df):
    X, y = get_data(messy_df, target="target", n_rows=20)
    assert len(X) == 20
    assert len(y) == 20


def test_get_data_n_rows_deterministic(messy_df):
    X1, _ = get_data(messy_df, target="target", n_rows=20)
    X2, _ = get_data(messy_df, target="target", n_rows=20)
    X3, _ = get_data(messy_df, target="target", n_rows=20, random_state=7)
    assert X1.equals(X2)
    assert not X1.equals(X3)


def test_get_data_drop_pattern():
    df = pd.DataFrame({
        "feat": [1.0, 2.0, 3.0, 4.0],
        "target_a": [1.0, 2.0, 3.0, 4.0],
        "target_b": [4.0, 3.0, 2.0, 1.0],
        "label": [0, 1, 0, 1],
    })
    X, _ = get_data(df, target="label", columns_to_drop=["target_*"])
    assert list(X.columns) == ["feat"]


def test_get_data_drop_pattern_never_drops_target():
    df = pd.DataFrame({
        "feat": [1.0, 2.0, 3.0, 4.0],
        "target_leak": [0.0, 1.0, 0.0, 1.0],
        "target": [0, 1, 0, 1],
    })
    X, y = get_data(df, target="target", columns_to_drop=["target*"])
    assert list(X.columns) == ["feat"]
    assert len(y) == 4


def test_get_data_drop_pattern_no_match(messy_df):
    with pytest.raises(ValueError, match="matched no columns"):
        get_data(messy_df, target="target", columns_to_drop=["nothing_*"])


def test_get_data_drop_mixed_exact_and_pattern(messy_df):
    X, _ = get_data(messy_df, target="target", columns_to_drop=["num_c", "cat_*"])
    assert "num_c" not in X.columns
    assert not any(column.startswith("cat_col") for column in X.columns)


def test_get_data_missing_target(messy_df):
    with pytest.raises(ValueError, match="not found"):
        get_data(messy_df, target="nonexistent")


def test_get_data_missing_columns(messy_df):
    with pytest.raises(ValueError, match="not found"):
        get_data(messy_df, target="target", columns_to_drop=["fake_col"])


def test_get_data_target_in_drop_list(messy_df):
    with pytest.raises(ValueError, match="columns_to_drop"):
        get_data(messy_df, target="target", columns_to_drop=["target"])


def test_get_data_bad_thresh(messy_df):
    with pytest.raises(ValueError, match="thresh"):
        get_data(messy_df, target="target", thresh=1.5)


def test_get_data_too_many_rows(messy_df):
    with pytest.raises(ValueError, match="remain after cleaning"):
        get_data(messy_df, target="target", n_rows=10_000)


def test_get_data_encodes_category_dtype():
    """v2 select_dtypes missed pandas category columns entirely."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": pd.Categorical(["x", "y", "x", "z"]),
        "target": [0, 1, 0, 1],
    })
    X, _ = get_data(df, target="target")
    assert list(X.columns) == ["num", "cat-x", "cat-y", "cat-z"]
    X_label, _ = get_data(df, target="target", encoding="label")
    assert np.issubdtype(X_label["cat"].dtype, np.integer)


def test_get_data_converts_datetime():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "when": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        "target": [0, 1, 0, 1],
    })
    X, _ = get_data(df, target="target")
    assert X["when"].dtype == np.int64


def test_get_data_encodes_string_target():
    """v2 returned string classification targets unencoded."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "target": ["sick", "healthy", "sick", "healthy"],
    })
    _, y = get_data(df, target="target")
    assert np.issubdtype(y.dtype, np.integer)
    assert set(y) == {0, 1}


def test_get_data_encodes_category_target():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "target": pd.Categorical(["a", "b", "a", "b"]),
    })
    _, y = get_data(df, target="target")
    assert np.issubdtype(y.dtype, np.integer)


def test_get_data_bool_target(messy_df):
    df = messy_df.copy()
    df["target"] = df["target"].astype(bool)
    _, y = get_data(df, target="target")
    assert np.issubdtype(y.dtype, np.integer)
