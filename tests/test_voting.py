"""Tests for weighted rank aggregation."""

import math

import numpy as np
import pytest

from featureranker.result import make_table
from featureranker.vote import voting


@pytest.fixture
def two_method_rankings():
    # Method A ranks: a=1, b=2, c=3, d=4; method B ranks: b=1, a=2, c=3, d=4.
    return {
        "A": make_table(("a", "b", "c", "d"), np.array([4.0, 3.0, 2.0, 1.0])),
        "B": make_table(("a", "b", "c", "d"), np.array([5.0, 10.0, 1.0, 0.5])),
    }


def _scores(table) -> dict[str, float]:
    return dict(zip(table["feature"], table["score"]))


def test_reciprocal_rank_hand_computed(two_method_rankings):
    scores = _scores(voting(two_method_rankings))
    assert scores["a"] == pytest.approx(1.0 + 1.0 / 2.0)
    assert scores["b"] == pytest.approx(1.0 / 2.0 + 1.0)
    assert scores["c"] == pytest.approx(2.0 / 3.0)
    assert scores["d"] == pytest.approx(1.0 / 2.0)


def test_tied_totals_order_by_name(two_method_rankings):
    table = voting(two_method_rankings)
    # a and b tie at 1.5: the stable tiebreak puts a first
    assert table["feature"].tolist()[:2] == ["a", "b"]


def test_borda_hand_computed(two_method_rankings):
    scores = _scores(voting(two_method_rankings, method="borda"))
    assert scores["a"] == pytest.approx(3.0 + 2.0)
    assert scores["b"] == pytest.approx(2.0 + 3.0)
    assert scores["c"] == pytest.approx(2.0)
    assert scores["d"] == pytest.approx(0.0)


def test_exponential_hand_computed(two_method_rankings):
    scores = _scores(voting(two_method_rankings, method="exponential"))
    assert scores["a"] == pytest.approx(1.0 + math.exp(-1.0 / 3.0))
    assert scores["c"] == pytest.approx(2.0 * math.exp(-2.0 / 3.0))
    assert scores["d"] == pytest.approx(2.0 * math.exp(-1.0))


def test_weights_scale_contributions(two_method_rankings):
    scores = _scores(voting(two_method_rankings, weights={"A": 2.0}))
    # B keeps the default weight 1.0
    assert scores["a"] == pytest.approx(2.0 * 1.0 + 1.0 / 2.0)
    assert scores["b"] == pytest.approx(2.0 * (1.0 / 2.0) + 1.0)


def test_unknown_weight_key_raises(two_method_rankings):
    with pytest.raises(ValueError, match="unknown methods"):
        voting(two_method_rankings, weights={"nope": 1.0})


def test_non_numeric_weight_raises(two_method_rankings):
    with pytest.raises(TypeError, match="must be a number"):
        voting(two_method_rankings, weights={"A": "heavy"})


def test_exact_ties_share_points():
    rankings = {"T": make_table(("x", "y", "z"), np.array([5.0, 5.0, 1.0]))}
    scores = _scores(voting(rankings))
    # x and y tie for ranks 1 and 2: both take the average rank 1.5
    assert scores["x"] == pytest.approx(1.0 / 1.5)
    assert scores["y"] == pytest.approx(1.0 / 1.5)
    assert scores["z"] == pytest.approx(1.0 / 3.0)


def test_invalid_method_raises(two_method_rankings):
    with pytest.raises(ValueError, match="voting method"):
        voting(two_method_rankings, method="plurality")


def test_empty_rankings_raise():
    with pytest.raises(ValueError, match="no rankings"):
        voting({})


def test_duplicate_feature_raises():
    table = make_table(("x", "x", "z"), np.array([3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="more than once"):
        voting({"T": table})
