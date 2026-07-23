import pytest

from dspy.teleprompt.darwin import (
    Checkpointable,
    GEPAConfig,
    GEPAStrategy,
    LMCallsBudget,
)


def test_lm_budget_tracks_evaluation_and_generation_domains():
    budget = LMCallsBudget(
        max_calls=10,
        evaluation_max_calls=6,
        generation_max_calls=4,
    )
    budget.spend_on_evaluation(None, {"phase": "full_evaluation", "examples": 5})
    budget.spend_on_generation(None)
    budget.spend_on_generation(None)

    remaining = budget._get_remaining()
    assert budget.evaluation_calls == 5
    assert budget.generation_calls == 2
    assert remaining["evaluation_calls"] == 1
    assert remaining["generation_calls"] == 2
    assert budget.can_spend("evaluation", 1)
    assert not budget.can_spend("evaluation", 2)


def test_lm_budget_rejects_unknown_domain():
    with pytest.raises(ValueError):
        LMCallsBudget(10).can_spend("reflection")


def test_non_billable_generation_failure_does_not_consume_budget():
    budget = LMCallsBudget(10)
    budget.spend_on_generation(
        None,
        {"type": "failed_mutation_attempt", "cost": 5, "billable": False},
    )

    assert budget.consumed_calls == 0
    assert budget.generation_calls == 0


def test_budget_state_can_be_restored_without_strategy_access_to_counters():
    budget = LMCallsBudget(
        max_calls=10,
        evaluation_max_calls=6,
        generation_max_calls=4,
    )
    budget.spend_on_evaluation(None, {"phase": "full_evaluation", "examples": 3})
    budget.spend_on_generation(None, {"cost": 2})

    restored = LMCallsBudget(
        max_calls=10,
        evaluation_max_calls=6,
        generation_max_calls=4,
    )
    restored.restore_state(budget.serialize_state())

    assert restored._get_remaining() == budget._get_remaining()
    assert isinstance(restored, Checkpointable)


def test_strategy_stops_when_the_active_budget_domain_is_exhausted():
    strategy = GEPAStrategy(GEPAConfig(max_lm_calls=10))
    strategy.budget = LMCallsBudget(
        max_calls=10, evaluation_max_calls=2, generation_max_calls=8
    )
    strategy.algorithm_state = "generate"
    strategy.budget.evaluation_calls = 2

    assert strategy.next_step() is False

    strategy = GEPAStrategy(GEPAConfig(max_lm_calls=10))
    strategy.budget = LMCallsBudget(
        max_calls=10, evaluation_max_calls=8, generation_max_calls=2
    )
    strategy.algorithm_state = "generate"
    strategy.budget.generation_calls = 2

    assert strategy.next_step() is False
