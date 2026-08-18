"""Hugging Face Hub integration: hub datasets as ranking-ready frames."""

import logging

import pandas as pd

from typing import Literal

from .data import get_data

logger = logging.getLogger(__name__)


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


def get_hf_data(
    path: str,
    target: str,
    split: str = "train",
    name: str | None = None,
    thresh: float = 0.8,
    columns_to_drop: list[str] | None = None,
    n_rows: int | None = None,
    random_state: int = 42,
    encoding: Literal["onehot", "label"] = "onehot",
    max_categories: int | None = 64,
    **load_kwargs: object,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download a Hub dataset and return (features, target) ready to rank.

    `target` names the label column and `columns_to_drop` excludes columns
    such as ids or free text; every remaining column becomes a feature. The
    frame goes through `get_data`, so cleaning, sampling, and categorical
    encoding behave exactly as documented there.
    """
    frame = load_hf_dataset(path, split=split, name=name, **load_kwargs)
    return get_data(
        frame,
        target=target,
        thresh=thresh,
        columns_to_drop=columns_to_drop,
        n_rows=n_rows,
        random_state=random_state,
        encoding=encoding,
        max_categories=max_categories,
    )
