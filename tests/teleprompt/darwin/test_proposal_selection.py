import pytest

from dspy.teleprompt.darwin import AllImprovements, BestImprovement, TopKImprovements


def test_all_improvements_preserves_batch():
    candidates = ["a", "b", "c"]
    assert AllImprovements().select(candidates, [1.0, 3.0, 2.0]) == candidates


def test_best_improvement_selects_largest_margin():
    assert BestImprovement().select(["a", "b", "c"], [1.0, 3.0, 2.0]) == ["b"]


def test_top_k_improvements_selects_in_descending_order():
    assert TopKImprovements(2).select(["a", "b", "c"], [1.0, 3.0, 2.0]) == ["b", "c"]


def test_top_k_requires_positive_k():
    with pytest.raises(ValueError):
        TopKImprovements(0)
