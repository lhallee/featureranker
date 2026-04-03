import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris


@pytest.fixture
def cancer_data():
    """Small subset of breast cancer dataset for fast classification tests."""
    data = load_breast_cancer(as_frame=True)
    df = data.data.merge(data.target, left_index=True, right_index=True)
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


@pytest.fixture
def diabetes_data():
    """Small subset of diabetes dataset for fast regression tests."""
    data = load_diabetes(as_frame=True)
    df = data.data.merge(data.target, left_index=True, right_index=True)
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


@pytest.fixture
def iris_data():
    """Iris dataset for multiclass classification tests."""
    data = load_iris(as_frame=True)
    df = data.data.merge(data.target, left_index=True, right_index=True)
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


@pytest.fixture
def messy_df():
    """DataFrame with NaNs, constant columns, and categorical features."""
    rng = np.random.RandomState(42)
    n = 50
    df = pd.DataFrame({
        "num_a": rng.randn(n),
        "num_b": rng.randn(n),
        "num_c": rng.randn(n),
        "cat_col": rng.choice(["A", "B", "C"], size=n),
        "constant": 1.0,
        "mostly_nan": [np.nan] * 45 + list(rng.randn(5)),
        "target": rng.randint(0, 2, size=n),
    })
    return df
