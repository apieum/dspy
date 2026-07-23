import pytest

from dspy.teleprompt.darwin import (
    BudgetEvent,
    BudgetExhaustedError,
    Checkpointable,
    LMCallsBudget,
)
from dspy.teleprompt.darwin.algorithms.gepa import GEPAConfig, GEPAStrategy


def test_lm_budget_tracks_evaluation_and_generation_domains():
    budget = LMCallsBudget(
        max_calls=10,
        evaluation_max_calls=6,
        generation_max_calls=4,
    )
    budget.spend(BudgetEvent("evaluation", 5, {"phase": "full_evaluation"}))
    budget.spend(BudgetEvent("generation"))
    budget.spend(BudgetEvent("generation"))

    assert budget.evaluation_calls == 5
    assert budget.generation_calls == 2
    assert budget.serialize_state()["evaluation_calls"] == 1
    assert budget.serialize_state()["generation_calls"] == 2
    budget.spend(BudgetEvent("evaluation"))
    with pytest.raises(BudgetExhaustedError):
        budget.spend(BudgetEvent("evaluation", 2))


def test_lm_budget_rejects_unknown_domain():
    with pytest.raises(ValueError):
        LMCallsBudget(10).spend(BudgetEvent("reflection"))


def test_non_billable_generation_failure_does_not_consume_budget():
    budget = LMCallsBudget(10)
    budget.spend(
        BudgetEvent(
            "generation", 5, {"type": "failed_mutation_attempt", "billable": False}
        )
    )

    assert budget.consumed_calls == 0
    assert budget.generation_calls == 0


def test_budget_state_can_be_restored_without_strategy_access_to_counters():
    budget = LMCallsBudget(
        max_calls=10,
        evaluation_max_calls=6,
        generation_max_calls=4,
    )
    budget.spend(BudgetEvent("evaluation", 3, {"phase": "full_evaluation"}))
    budget.spend(BudgetEvent("generation", 2))

    restored = LMCallsBudget.restore_state(budget.serialize_state())

    assert restored.serialize_state() == budget.serialize_state()
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
