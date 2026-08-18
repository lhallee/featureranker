"""Dataset preparation: cleaning, sampling, and encoding before ranking."""

import fnmatch
import logging
import re

import numpy as np
import pandas as pd

from typing import Literal

from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-word characters in column names with underscores."""
    df = df.copy()
    df.columns = [re.sub(r"[^\w]", "_", str(column)) for column in df.columns]
    return df


def view_data(df: pd.DataFrame) -> pd.Series:
    """Percentage of missing values per column, only columns that have any."""
    nan_pct = (df.isna().sum() / len(df) * 100).round(1)
    nan_pct = nan_pct[nan_pct > 0]
    if nan_pct.empty:
        logger.info("No missing values in the dataset.")
    for column, pct in nan_pct.items():
        logger.info("Column %s is %.1f%% missing.", column, pct)
    return nan_pct


def _resolve_columns_to_drop(
    df: pd.DataFrame, target: str, columns_to_drop: list[str] | None
) -> list[str]:
    """Expand exact names and glob patterns ("target_*") into drop columns.

    Exact names must exist and may not name the target. Entries containing
    a wildcard (* ? [) match by fnmatch against every column except the
    target, so a pattern can never drop the label; a pattern matching
    nothing raises to catch typos.
    """
    dropped: list[str] = []
    missing: list[str] = []
    for entry in columns_to_drop or ():
        if any(wildcard in str(entry) for wildcard in "*?["):
            matches = [
                column for column in df.columns
                if column != target and fnmatch.fnmatchcase(str(column), str(entry))
            ]
            if not matches:
                raise ValueError(f"Pattern {entry!r} matched no columns.")
            dropped.extend(column for column in matches if column not in dropped)
        elif entry == target:
            raise ValueError(
                f"Target column {target!r} is also listed in columns_to_drop."
            )
        elif entry not in df.columns:
            missing.append(entry)
        elif entry not in dropped:
            dropped.append(entry)
    if missing:
        raise ValueError(f"Columns to drop not found: {sorted(missing)}.")
    return dropped


def _is_categorical(values: pd.Series) -> bool:
    """Object, string, and category dtypes take categorical encoding."""
    return values.dtype == object or isinstance(
        values.dtype, (pd.StringDtype, pd.CategoricalDtype)
    )


def _one_hot(values: pd.Series, column: str) -> pd.DataFrame:
    """Expand one categorical column into 0/1 sub-features named column-value."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.remove_unused_categories()
    return pd.get_dummies(values, prefix=column, prefix_sep="-", dtype=np.int8)  # (n, k)


def _encode_features(
    clean: pd.DataFrame,
    feature_columns: list[str],
    encoding: str,
    max_categories: int | None,
) -> pd.DataFrame:
    """Encode temporal, boolean, and categorical feature columns.

    With encoding="onehot", each categorical column expands into one 0/1
    sub-feature per unique value in the data, named "{column}-{value}" and
    injected at the parent column's position; columns whose cardinality
    exceeds max_categories fall back to label encoding. With
    encoding="label", categorical columns are label-encoded in place. Under
    both settings booleans become single 0/1 columns and datetime/timedelta
    columns become int64 nanoseconds.
    """
    features = clean[feature_columns]
    temporal = set(
        features.select_dtypes(include=["datetime", "datetimetz", "timedelta"]).columns
    )
    boolean = set(features.select_dtypes(include=["bool"]).columns)

    pieces: list[pd.Series | pd.DataFrame] = []
    for column in feature_columns:
        values = clean[column]  # (n,)
        if column in temporal:
            pieces.append(values.astype("int64"))
            logger.info("Converted temporal column %s to int64 nanoseconds.", column)
        elif column in boolean:
            pieces.append(values.astype(np.int8))
            logger.info("Converted boolean column %s to 0/1.", column)
        elif _is_categorical(values):
            n_categories = values.nunique(dropna=False)
            if encoding == "onehot" and (
                max_categories is None or n_categories <= max_categories
            ):
                sub_features = _one_hot(values, str(column))  # (n, k)
                pieces.append(sub_features)
                logger.info(
                    "One-hot encoded column %s into %d sub-features.",
                    column, sub_features.shape[1],
                )
            else:
                encoded = LabelEncoder().fit_transform(values)  # (n,)
                pieces.append(pd.Series(encoded, index=values.index, name=column))
                if encoding == "onehot":
                    logger.info(
                        "Label encoded column %s: %d unique values exceed "
                        "max_categories=%d.", column, n_categories, max_categories,
                    )
                else:
                    logger.info("Label encoded column %s.", column)
        else:
            pieces.append(values)

    X = pd.concat(pieces, axis=1)  # (n, p_encoded)
    duplicates = X.columns[X.columns.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"Encoding produced duplicate column names: {duplicates}. "
            "Rename the conflicting columns before calling get_data."
        )
    return X


def _encode_target(y: pd.Series) -> pd.Series:
    """Encode a non-numeric target; numeric targets pass through."""
    if y.dtype == bool:
        return y.astype(int)
    if y.dtype == object or isinstance(y.dtype, (pd.StringDtype, pd.CategoricalDtype)):
        encoded = LabelEncoder().fit_transform(y)  # (n,)
        logger.info("Encoded target %s.", y.name)
        return pd.Series(encoded, index=y.index, name=y.name)
    return y


def get_data(
    df: pd.DataFrame,
    target: str,
    thresh: float = 0.8,
    columns_to_drop: list[str] | None = None,
    n_rows: int | None = None,
    random_state: int = 42,
    encoding: Literal["onehot", "label"] = "onehot",
    max_categories: int | None = 64,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a raw frame for ranking and return (features, target).

    Steps, in order: drop requested columns, drop feature columns with less
    than `thresh` fraction of values present, drop rows with remaining
    missing values, optionally shuffle-sample n_rows, drop constant columns,
    then encode features and a non-numeric target. columns_to_drop entries
    are exact names or glob patterns ("target_*" drops every match; a
    pattern never drops the target itself). Categorical feature columns
    one-hot expand into "{column}-{value}" sub-features by default;
    encoding="label" keeps them as single label-encoded columns, and
    columns with more than max_categories unique values fall back to label
    encoding (max_categories=None one-hot encodes at any cardinality).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if not 0.0 < thresh <= 1.0:
        raise ValueError(f"thresh must be in (0, 1], got {thresh}.")
    if encoding not in ("onehot", "label"):
        raise ValueError(f"Unknown encoding {encoding!r}. Valid: 'onehot', 'label'.")
    if max_categories is not None and max_categories < 2:
        raise ValueError(f"max_categories must be None or >= 2, got {max_categories}.")
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in the DataFrame.")

    dropped = _resolve_columns_to_drop(df, target, columns_to_drop)
    if dropped:
        logger.info("Dropping columns: %s.", dropped)

    feature_columns = [
        column for column in df.columns if column != target and column not in dropped
    ]
    present_counts = df[feature_columns].notna().sum()  # (p,)
    kept_columns = [
        column
        for column in feature_columns
        if present_counts[column] >= int(thresh * len(df))
    ]
    logger.info(
        "Kept %d of %d feature columns with at least %.0f%% present values.",
        len(kept_columns),
        len(feature_columns),
        thresh * 100,
    )

    complete_rows = df[kept_columns].notna().all(axis=1) & df[target].notna()  # (n,)
    clean = df.loc[complete_rows, kept_columns + [target]].copy()

    if n_rows is not None:
        if n_rows > len(clean):
            raise ValueError(
                f"Requested {n_rows} rows but only {len(clean)} remain after cleaning."
            )
        clean = clean.sample(n=n_rows, random_state=random_state)
        logger.info("Shuffled and sampled %d rows.", n_rows)
    clean = clean.reset_index(drop=True)

    constant_columns = [
        column for column in kept_columns if clean[column].nunique(dropna=False) == 1
    ]
    if constant_columns:
        clean = clean.drop(columns=constant_columns)
        kept_columns = [c for c in kept_columns if c not in constant_columns]
        logger.info("Removed constant columns: %s.", constant_columns)

    X = _encode_features(clean, kept_columns, encoding, max_categories)  # (n, p_encoded)
    y = _encode_target(clean[target])  # (n,)
    return X, y
