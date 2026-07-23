"""Focused checks for behavior shared with the reference GEPA engine."""

import random

import dspy

from dspy.teleprompt.darwin import (
    ImprovementOrEqualAcceptance,
    Metric,
    StrictImprovementAcceptance,
)
from dspy.teleprompt.darwin.algorithms.gepa import (
    GEPACandidate as Candidate, ParentFastCompare, ParetoFrontier, GEPAConfig,
)
from dspy.teleprompt.darwin.budget import LMCallsBudget
from dspy.teleprompt.darwin.evaluation.cache import EvaluationCache
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
    selector = ParetoFrontier(GEPAConfig())
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
    selector = ParetoFrontier(GEPAConfig())
    dominant = candidate([1.0, 1.0])
    dominated = candidate([1.0, 0.0])
    selector.update_score("example-0", dominant, Metric(1.0, id="example-0"))
    selector.update_score("example-1", dominant, Metric(1.0, id="example-1"))
    selector.update_score("example-0", dominated, Metric(1.0, id="example-0"))
    selector.update_score("example-1", dominated, Metric(0.0, id="example-1"))

    parents = selector.promote(Survivors(iteration=0))

    assert dominant in parents
    assert dominated not in parents


def test_merge_accepts_tie_with_best_parent_but_mutation_does_not():
    ancestor = Candidate(dspy.Predict("question -> answer"))
    parent1 = Candidate(ancestor.module.deepcopy(), parents=[ancestor])
    parent2 = Candidate(ancestor.module.deepcopy(), parents=[ancestor])
    merged = Candidate(
        ancestor.module.deepcopy(),
        parents=[parent1, parent2, ancestor],
        creation_metadata={"merge_type": "system_aware", "ancestor_candidate": ancestor},
    )
    example = dspy.Example(question="q", answer="a")
    evaluator = ParentFastCompare(
        config=GEPAConfig(fitness_function=lambda *_: Metric(0.0)),
        evaluation_cache=EvaluationCache(),
        assessor=lambda *_: Metric(0.0),
        minibatch_data=[example],
    )
    values = {id(merged): 0.8, id(parent1): 0.8, id(parent2): 0.7}
    evaluator._evaluate = lambda candidate, examples, persist_scores=False: [
        Metric(values[id(candidate)], id="example")
    ]

    accepted, _, improvement = evaluator._validate_on_minibatch(
        merged, LMCallsBudget(max_calls=10)
    )

    assert accepted
    assert improvement == 0.0
