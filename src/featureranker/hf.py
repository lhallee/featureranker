"""Hugging Face Hub integration: hub datasets as ranking-ready frames."""

import logging

import pandas as pd

from dataclasses import dataclass
from typing import Literal

from .data import get_data

logger = logging.getLogger(__name__)

_VALID_NAMES = ("validation", "valid", "val", "dev")
_TEST_NAMES = ("test",)


@dataclass(frozen=True)
class DataSplits:
    """Prepared (features, target) pairs for the splits a dataset provides.

    Cleaning and encoding are computed jointly over every split, so the
    feature columns always match across them. Best practice: rank and fit
    on train, compare and tune on valid, report the final result on test.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame | None = None
    y_valid: pd.Series | None = None
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None

    @property
    def valid(self) -> tuple[pd.DataFrame, pd.Series] | None:
        """The validation pair as (X, y), or None when the split is absent."""
        if self.X_valid is None:
            return None
        return self.X_valid, self.y_valid

    @property
    def test(self) -> tuple[pd.DataFrame, pd.Series] | None:
        """The test pair as (X, y), or None when the split is absent."""
        if self.X_test is None:
            return None
        return self.X_test, self.y_test

    def __repr__(self) -> str:
        sizes = [f"train={len(self.y_train)}"]
        if self.y_valid is not None:
            sizes.append(f"valid={len(self.y_valid)}")
        if self.y_test is not None:
            sizes.append(f"test={len(self.y_test)}")
        return f"DataSplits({', '.join(sizes)}, n_features={self.X_train.shape[1]})"


def hf_login(token: str | None = None) -> None:
    """Authenticate with the Hugging Face Hub for private or gated datasets.

    Without a token this opens the interactive Hub prompt; with a token it
    stores the credential the same way `hf auth login` does. Public datasets
    need no login.
    """
    from huggingface_hub import login

    login(token=token)


def load_hf_dataset(
    path: str,
    split: str = "train",
    name: str | None = None,
    **load_kwargs: object,
) -> pd.DataFrame:
    """Download one split of a Hub dataset and return it as a DataFrame.

    `path` is the Hub id, for example "scikit-learn/adult-census-income";
    `name` selects a configuration for multi-config datasets. Extra keyword
    arguments pass through to `datasets.load_dataset` (revision, data_files,
    token, ...).
    """
    # datasets pulls in pyarrow and probes optional ML backends on import;
    # importing it here keeps `import featureranker` fast
    from datasets import load_dataset

    rows = load_dataset(path, name=name, split=split, **load_kwargs)
    if isinstance(rows, dict):
        raise ValueError(
            f"Split {split!r} of {path!r} resolved to multiple splits "
            f"{list(rows)}; pass exactly one split name."
        )
    frame = rows.to_pandas()  # (n, columns)
    logger.info(
        "Loaded %s split %s: %d rows, %d columns.",
        path, split, len(frame), frame.shape[1],
    )
    return frame


def _resolve_extra_splits(
    path: str,
    name: str | None,
    split: str,
    valid_split: str | None,
    test_split: str | None,
) -> tuple[str | None, str | None]:
    """Resolve "auto" valid/test split requests against the dataset's splits."""
    if valid_split == "auto" or test_split == "auto":
        from datasets import get_dataset_split_names

        available = {s.lower(): s for s in get_dataset_split_names(path, name)}
        if valid_split == "auto":
            valid_split = next(
                (available[n] for n in _VALID_NAMES if n in available), None
            )
        if test_split == "auto":
            test_split = next(
                (available[n] for n in _TEST_NAMES if n in available), None
            )
    if valid_split == split:
        valid_split = None
    if test_split == split or (test_split is not None and test_split == valid_split):
        test_split = None
    return valid_split, test_split


def get_hf_data(
    path: str,
    target: str,
    split: str = "train",
    valid_split: str | None = "auto",
    test_split: str | None = "auto",
    name: str | None = None,
    thresh: float = 0.8,
    columns_to_drop: list[str] | None = None,
    n_rows: int | None = None,
    random_state: int = 42,
    encoding: Literal["onehot", "label"] = "onehot",
    max_categories: int | None = 64,
    **load_kwargs: object,
) -> tuple[pd.DataFrame, pd.Series] | DataSplits:
    """Download a Hub dataset and prepare it for ranking, split-aware.

    `target` names the label column and `columns_to_drop` excludes columns
    (exact names or glob patterns like "target_*"); every remaining column
    becomes a feature. With only a train split the return is (X, y) as
    before. When the dataset also provides validation or test splits (found
    automatically under "auto", named explicitly, or disabled with None),
    the return is a `DataSplits`: cleaning and encoding run jointly over
    all splits so the feature columns match, rows are cleaned per split,
    and `n_rows` samples the train split only. Rank and fit on train, tune
    on valid, report on test.
    """
    valid_split, test_split = _resolve_extra_splits(
        path, name, split, valid_split, test_split
    )
    frames = [load_hf_dataset(path, split=split, name=name, **load_kwargs)]
    for extra in (valid_split, test_split):
        if extra is not None:
            frames.append(load_hf_dataset(path, split=extra, name=name, **load_kwargs))

    if len(frames) == 1:
        return get_data(
            frames[0], target=target, thresh=thresh,
            columns_to_drop=columns_to_drop, n_rows=n_rows,
            random_state=random_state, encoding=encoding,
            max_categories=max_categories,
        )

    combined = pd.concat(frames, ignore_index=True)  # (n_total, columns)
    X_all, y_all = get_data(
        combined, target=target, thresh=thresh, columns_to_drop=columns_to_drop,
        n_rows=None, random_state=random_state, encoding=encoding,
        max_categories=max_categories,
    )

    pairs: list[tuple[pd.DataFrame, pd.Series]] = []
    start = 0
    for frame in frames:
        end = start + len(frame)
        rows = (X_all.index >= start) & (X_all.index < end)  # (n_clean,)
        pairs.append((X_all[rows], y_all[rows]))
        start = end

    X_train, y_train = pairs[0]
    if n_rows is not None:
        if n_rows > len(X_train):
            raise ValueError(
                f"Requested {n_rows} rows but only {len(X_train)} train rows "
                "remain after cleaning."
            )
        X_train = X_train.sample(n=n_rows, random_state=random_state)
        y_train = y_train.loc[X_train.index]
        logger.info("Shuffled and sampled %d train rows.", n_rows)

    splits = {"X_train": X_train, "y_train": y_train}
    extra_pairs = iter(pairs[1:])
    for key, extra in (("valid", valid_split), ("test", test_split)):
        if extra is not None:
            splits[f"X_{key}"], splits[f"y_{key}"] = next(extra_pairs)
    prepared = DataSplits(**splits)
    logger.info("Prepared %r.", prepared)
    return prepared
