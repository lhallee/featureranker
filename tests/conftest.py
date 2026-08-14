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


def _named_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])


def _shifted_classification(
    n_samples: int, n_features: int, n_informative: int, n_classes: int, shift: float
) -> tuple[pd.DataFrame, pd.Series]:
    """Noise features plus informative features mean-shifted per class.

    Every informative feature gets a marginal class separation of `shift`
    standard deviations per class step, so univariate and multivariate
    methods alike must detect all of them.
    """
    rng = np.random.default_rng(42)
    y = rng.integers(0, n_classes, size=n_samples)  # (n,)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))  # (n, p)
    X[:, :n_informative] += shift * y[:, None]  # (n, k) shifted
    return _named_frame(X), pd.Series(y, name="target")


@pytest.fixture(scope="session")
def synthetic_classification():
    """300 x 12 binary frame; feat_0..feat_3 informative by mean shift."""
    return _shifted_classification(
        n_samples=300, n_features=12, n_informative=4, n_classes=2, shift=2.0
    )


@pytest.fixture(scope="session")
def synthetic_multiclass():
    """300 x 12 three-class frame; feat_0..feat_3 informative by mean shift."""
    return _shifted_classification(
        n_samples=300, n_features=12, n_informative=4, n_classes=3, shift=1.5
    )


@pytest.fixture(scope="session")
def synthetic_regression():
    """300 x 12 regression frame; shuffle=False makes feat_0..feat_2 informative."""
    from sklearn.datasets import make_regression

    X, y, coef = make_regression(
        n_samples=300, n_features=12, n_informative=3, shuffle=False,
        coef=True, noise=1.0, random_state=42,
    )
    return _named_frame(X), pd.Series(y, name="target"), coef


@pytest.fixture(scope="session")
def wide_data():
    """60 x 300 binary frame (p > n); feat_0..feat_9 informative by mean shift."""
    return _shifted_classification(
        n_samples=60, n_features=300, n_informative=10, n_classes=2, shift=2.0
    )


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
