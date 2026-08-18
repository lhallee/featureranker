"""Tests for the Hugging Face integration, with the hub mocked out."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from featureranker.hf import get_hf_data, hf_login, load_hf_dataset


class FakeSplit:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


@pytest.fixture
def hub_frame():
    rng = np.random.RandomState(0)
    n = 40
    return pd.DataFrame({
        "sepal": rng.randn(n),
        "petal": rng.randn(n),
        "color": rng.choice(["blue", "red"], size=n),
        "note": [f"free text {i}" for i in range(n)],
        "label": rng.randint(0, 2, size=n),
    })


@pytest.fixture
def fake_hub(monkeypatch, hub_frame):
    """Install a fake datasets module and record every load_dataset call."""
    calls = {}

    def load_dataset(path, name=None, split=None, **kwargs):
        calls.update(path=path, name=name, split=split, kwargs=kwargs)
        if split is None:
            return {"train": FakeSplit(hub_frame), "test": FakeSplit(hub_frame)}
        return FakeSplit(hub_frame)

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_load_hf_dataset_returns_frame(fake_hub, hub_frame):
    frame = load_hf_dataset("org/dataset")
    assert frame.equals(hub_frame)
    assert fake_hub["path"] == "org/dataset"
    assert fake_hub["split"] == "train"
    assert fake_hub["name"] is None


def test_load_hf_dataset_passes_config_and_kwargs(fake_hub):
    load_hf_dataset("org/dataset", split="test", name="coarse", revision="abc")
    assert fake_hub["split"] == "test"
    assert fake_hub["name"] == "coarse"
    assert fake_hub["kwargs"] == {"revision": "abc"}


def test_load_hf_dataset_rejects_multi_split(fake_hub):
    with pytest.raises(ValueError, match="multiple splits"):
        load_hf_dataset("org/dataset", split=None)


def test_get_hf_data_end_to_end(fake_hub):
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


def test_hf_login_forwards_token(monkeypatch):
    calls = {}
    module = types.ModuleType("huggingface_hub")
    module.login = lambda token=None: calls.update(token=token)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    hf_login(token="hf_test")
    assert calls["token"] == "hf_test"
