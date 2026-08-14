"""Dataset preparation: cleaning, sampling, and encoding before ranking."""

import logging
import re

import pandas as pd

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


def _encode_features(clean: pd.DataFrame, feature_columns: list[str]) -> None:
    """Encode categorical and temporal feature columns in place."""
    features = clean[feature_columns]
    temporal = features.select_dtypes(
        include=["datetime", "datetimetz", "timedelta"]
    ).columns
    for column in temporal:
        clean[column] = clean[column].astype("int64")
        logger.info("Converted temporal column %s to int64 nanoseconds.", column)

    categorical = features.select_dtypes(
        include=["object", "string", "bool", "category"]
    ).columns
    for column in categorical:
        clean[column] = LabelEncoder().fit_transform(clean[column])
        logger.info("Encoded column %s.", column)


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
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a raw frame for ranking and return (features, target).

    Steps, in order: drop requested columns, drop feature columns with less
    than `thresh` fraction of values present, drop rows with remaining
    missing values, optionally shuffle-sample n_rows, drop constant columns,
    then encode categorical/temporal features and a non-numeric target.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if not 0.0 < thresh <= 1.0:
        raise ValueError(f"thresh must be in (0, 1], got {thresh}.")
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in the DataFrame.")

    dropped = list(columns_to_drop or ())
    missing_drops = set(dropped) - set(df.columns)
    if missing_drops:
        raise ValueError(f"Columns to drop not found: {sorted(missing_drops)}.")
    if target in dropped:
        raise ValueError(f"Target column {target!r} is also listed in columns_to_drop.")
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

    _encode_features(clean, kept_columns)
    X = clean[kept_columns]
    y = _encode_target(clean[target])
    return X, y
