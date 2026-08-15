"""Tests for the RankingResult container and shared table schema."""

import numpy as np
import pytest

import featureranker

from featureranker.result import RankingResult, make_table


def _small_result() -> RankingResult:
    table_mi = make_table(("a", "b", "c"), np.array([3.0, 1.0, 2.0]))
    table_f = make_table(("a", "b", "c"), np.array([0.5, 0.5, 9.0]))
    return RankingResult(
        task="classification",
        feature_names=("a", "b", "c"),
        n_samples=10,
        n_features=3,
        rankings={"mi": table_mi, "f_test": table_f},
        diagnostics={"mi": {"n_neighbors": 3}, "f_test": {}},
        classes=(0, 1),
        random_state=42,
        version=featureranker.__version__,
    )


def test_make_table_sorts_best_first():
    table = make_table(("a", "b", "c"), np.array([1.0, 3.0, 2.0]))
    assert list(table.columns) == ["feature", "score"]
    assert table["feature"].tolist() == ["b", "c", "a"]
    assert table["score"].tolist() == [3.0, 2.0, 1.0]


def test_make_table_breaks_ties_by_name():
    table = make_table(("z", "m", "a"), np.array([1.0, 1.0, 1.0]))
    assert table["feature"].tolist() == ["a", "m", "z"]


def test_make_table_length_mismatch():
    with pytest.raises(ValueError, match="feature names"):
        make_table(("a", "b"), np.array([1.0]))


def test_methods_property():
    assert _small_result().methods == ("mi", "f_test")


def test_score_matrix_alignment():
    S = _small_result().score_matrix()
    assert list(S.index) == ["a", "b", "c"]
    assert list(S.columns) == ["mi", "f_test"]
    assert S.loc["a", "mi"] == 3.0
    assert S.loc["c", "f_test"] == 9.0


def test_rank_matrix_averages_ties():
    R = _small_result().rank_matrix()
    assert R.loc["a", "mi"] == 1.0
    assert R.loc["b", "mi"] == 3.0
    # f_test scores tie at 0.5 for a and b: both take rank (2 + 3) / 2
    assert R.loc["a", "f_test"] == 2.5
    assert R.loc["b", "f_test"] == 2.5
    assert R.loc["c", "f_test"] == 1.0


def test_save_load_round_trip(tmp_path):
    result = _small_result()
    path = tmp_path / "result.joblib"
    result.save(path)
    loaded = RankingResult.load(path)
    assert loaded.equals(result)
    assert loaded.diagnostics["mi"] == {"n_neighbors": 3}


def test_load_warns_on_version_mismatch(tmp_path):
    result = _small_result()
    stale = RankingResult(
        task=result.task,
        feature_names=result.feature_names,
        n_samples=result.n_samples,
        n_features=result.n_features,
        rankings=result.rankings,
        diagnostics=result.diagnostics,
        classes=result.classes,
        random_state=result.random_state,
        version="0.0.1",
    )
    path = tmp_path / "stale.joblib"
    stale.save(path)
    with pytest.warns(UserWarning, match="0.0.1"):
        RankingResult.load(path)


def test_load_rejects_corrupt_file(tmp_path):
    path = tmp_path / "garbage.joblib"
    path.write_bytes(b"not a joblib payload")
    with pytest.raises(ValueError, match="Could not load"):
        RankingResult.load(path)


def test_load_rejects_wrong_object(tmp_path):
    import joblib

    path = tmp_path / "dict.joblib"
    joblib.dump({"feature": "a"}, path)
    with pytest.raises(ValueError, match="does not contain"):
        RankingResult.load(path)


def test_equals_detects_score_changes():
    result = _small_result()
    other = _small_result()
    assert result.equals(other)
    other.rankings["mi"].loc[0, "score"] = 99.0
    assert not result.equals(other)


def test_repr_is_compact():
    text = repr(_small_result())
    assert "classification" in text
    assert "n_features=3" in text
    assert "score" not in text
