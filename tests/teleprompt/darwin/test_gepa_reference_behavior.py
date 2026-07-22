"""Focused checks for behavior shared with the reference GEPA engine."""

import random

import dspy

from dspy.teleprompt.darwin import (
    Candidate,
    ImprovementOrEqualAcceptance,
    Metric,
    ParetoFrontier,
    StrictImprovementAcceptance,
)
from dspy.teleprompt.darwin.data.cohort import Survivors


def candidate(score_values):
    result = Candidate(dspy.Predict("question -> answer"))
    result.scores = [Metric(value, id=f"example-{idx}") for idx, value in enumerate(score_values)]
    return result


def test_gepa_default_acceptance_requires_strict_improvement():
    criterion = StrictImprovementAcceptance()

    assert criterion.should_accept([0.8, 1.0], [0.8, 0.9])
    assert not criterion.should_accept([0.8, 0.9], [0.8, 0.9])
    assert not criterion.should_accept([0.8, 0.8], [0.8, 0.9])


def test_equal_acceptance_is_explicitly_available():
    criterion = ImprovementOrEqualAcceptance()

    assert criterion.should_accept([0.8, 0.9], [0.8, 0.9])
    assert not criterion.should_accept([0.8, 0.8], [0.8, 0.9])


def test_pareto_selection_is_seeded_and_frequency_weighted():
    selector = ParetoFrontier()
    frequent = candidate([1.0, 1.0, 1.0])
    specialist = candidate([1.0, 0.0, 0.0])
    selector.update_score("example-0", frequent, Metric(1.0, id="example-0"))
    selector.update_score("example-1", frequent, Metric(1.0, id="example-1"))
    selector.update_score("example-2", frequent, Metric(1.0, id="example-2"))
    selector.update_score("example-0", specialist, Metric(1.0, id="example-0"))

    first = selector.select_candidate(rng=random.Random(7))
    second = selector.select_candidate(rng=random.Random(7))
    assert first is second
    assert first in {frequent, specialist}
    assert selector.select_candidate(strategy="current_best") is frequent


def test_pareto_parent_frontier_removes_globally_dominated_candidate():
    selector = ParetoFrontier()
    dominant = candidate([1.0, 1.0])
    dominated = candidate([1.0, 0.0])
    selector.update_score("example-0", dominant, Metric(1.0, id="example-0"))
    selector.update_score("example-1", dominant, Metric(1.0, id="example-1"))
    selector.update_score("example-0", dominated, Metric(1.0, id="example-0"))
    selector.update_score("example-1", dominated, Metric(0.0, id="example-1"))

    parents = selector.promote(Survivors(iteration=0))

    assert dominant in parents
    assert dominated not in parents
