"""Tests for the Hugging Face integration, with the hub mocked out."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from featureranker.hf import DataSplits, get_hf_data, hf_login, load_hf_dataset


class FakeSplit:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


def _frame(n: int, colors: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "sepal": rng.randn(n),
        "petal": rng.randn(n),
        "color": rng.choice(colors, size=n),
        "note": [f"free text {i}" for i in range(n)],
        "label": rng.randint(0, 2, size=n),
    })


@pytest.fixture
def hub_frame():
    return _frame(40, ["blue", "red"], seed=0)


def _install_hub(monkeypatch, split_frames: dict[str, pd.DataFrame]) -> dict:
    """Install a fake datasets module serving the given splits."""
    calls: dict[str, object] = {"loaded": []}

    def load_dataset(path, name=None, split=None, **kwargs):
        calls.update(path=path, name=name, kwargs=kwargs)
        calls["loaded"].append(split)
        if split is None:
            return {key: FakeSplit(frame) for key, frame in split_frames.items()}
        return FakeSplit(split_frames[split])

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    module.get_dataset_split_names = lambda path, name=None: list(split_frames)
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


@pytest.fixture
def fake_hub(monkeypatch, hub_frame):
    return _install_hub(monkeypatch, {"train": hub_frame})


@pytest.fixture
def fake_hub_with_test(monkeypatch, hub_frame):
    return _install_hub(monkeypatch, {
        "train": hub_frame,
        "test": _frame(20, ["blue", "red", "green"], seed=1),
    })


@pytest.fixture
def fake_hub_all_splits(monkeypatch, hub_frame):
    return _install_hub(monkeypatch, {
        "train": hub_frame,
        "validation": _frame(15, ["blue", "red"], seed=2),
        "test": _frame(20, ["blue", "red"], seed=3),
    })


def test_load_hf_dataset_returns_frame(fake_hub, hub_frame):
    frame = load_hf_dataset("org/dataset")
    assert frame.equals(hub_frame)
    assert fake_hub["path"] == "org/dataset"
    assert fake_hub["loaded"] == ["train"]
    assert fake_hub["name"] is None


def test_load_hf_dataset_passes_config_and_kwargs(fake_hub):
    load_hf_dataset("org/dataset", split="train", name="coarse", revision="abc")
    assert fake_hub["name"] == "coarse"
    assert fake_hub["kwargs"] == {"revision": "abc"}


def test_load_hf_dataset_rejects_multi_split(fake_hub):
    with pytest.raises(ValueError, match="multiple splits"):
        load_hf_dataset("org/dataset", split=None)


def test_get_hf_data_train_only_returns_pair(fake_hub):
    X, y = get_hf_data("org/dataset", target="label", columns_to_drop=["note"])
    assert "note" not in X.columns
    assert "label" not in X.columns
    assert {"color-blue", "color-red"} <= set(X.columns)
    assert len(X) == len(y)
    assert np.issubdtype(y.dtype, np.integer)


def test_get_hf_data_forwards_get_data_options(fake_hub):
    X, _ = get_hf_data(
        "org/dataset", target="label",
        columns_to_drop=["note"], encoding="label", n_rows=10,
    )
    assert "color" in X.columns
    assert len(X) == 10


def test_get_hf_data_auto_test_split(fake_hub_with_test):
    splits = get_hf_data("org/dataset", target="label", columns_to_drop=["note"])
    assert isinstance(splits, DataSplits)
    assert len(splits.y_train) == 40
    assert len(splits.y_test) == 20
    assert splits.valid is None
    assert list(splits.X_train.columns) == list(splits.X_test.columns)
    assert splits.test == (splits.X_test, splits.y_test)


def test_get_hf_data_joint_encoding_covers_test_only_categories(fake_hub_with_test):
    """A category seen only in test still becomes a column in every split."""
    splits = get_hf_data("org/dataset", target="label", columns_to_drop=["note"])
    assert "color-green" in splits.X_train.columns
    assert splits.X_train["color-green"].sum() == 0
    assert splits.X_test["color-green"].sum() > 0


def test_get_hf_data_auto_valid_and_test(fake_hub_all_splits):
    splits = get_hf_data("org/dataset", target="label", columns_to_drop=["note"])
    assert len(splits.y_valid) == 15
    assert len(splits.y_test) == 20
    assert "valid=15" in repr(splits)


def test_get_hf_data_n_rows_samples_train_only(fake_hub_with_test):
    splits = get_hf_data(
        "org/dataset", target="label", columns_to_drop=["note"], n_rows=10
    )
    assert len(splits.y_train) == 10
    assert len(splits.y_test) == 20


def test_get_hf_data_splits_disabled(fake_hub_with_test):
    X, y = get_hf_data(
        "org/dataset", target="label", columns_to_drop=["note"],
        valid_split=None, test_split=None,
    )
    assert len(y) == 40


def test_get_hf_data_explicit_test_split(fake_hub_with_test):
    splits = get_hf_data(
        "org/dataset", target="label", columns_to_drop=["note"],
        valid_split=None, test_split="test",
    )
    assert isinstance(splits, DataSplits)
    assert fake_hub_with_test["loaded"] == ["train", "test"]


def test_hf_login_forwards_token(monkeypatch):
    calls = {}
    module = types.ModuleType("huggingface_hub")
    module.login = lambda token=None: calls.update(token=token)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    hf_login(token="hf_test")
    assert calls["token"] == "hf_test"
