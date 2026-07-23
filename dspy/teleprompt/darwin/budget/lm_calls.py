"""LM calls budget implementation."""

import logging
from typing import Any, Mapping
from .budget import (
    Budget,
    BudgetEvent,
    BudgetExhaustedError,
    configured_limit,
)

logger = logging.getLogger(__name__)


class LMCallsBudget(Budget):
    """Budget that tracks LLM API calls."""

    def __init__(
        self,
        max_calls: int | None = None,
        evaluation_max_calls: int | None = None,
        generation_max_calls: int | None = None,
        config=None,
    ):
        self.max_calls = configured_limit(
            max_calls,
            config,
            "max_lm_calls",
            "LMCallsBudget requires max_calls or a configuration providing it",
        )
        configured_evaluation = getattr(config, "max_evaluation_calls", None)
        configured_generation = getattr(config, "max_generation_calls", None)
        self.evaluation_max_calls = int(
            evaluation_max_calls
            if evaluation_max_calls is not None
            else configured_evaluation
            if configured_evaluation is not None
            else self.max_calls
        )
        self.generation_max_calls = int(
            generation_max_calls
            if generation_max_calls is not None
            else configured_generation
            if configured_generation is not None
            else self.max_calls
        )
        self.consumed_calls = 0
        self.evaluation_calls = 0
        self.generation_calls = 0
    def reconcile(self, strategy) -> None:
        if (
            self.consumed_calls >= self.max_calls
            or self.evaluation_calls >= self.evaluation_max_calls
            or self.generation_calls >= self.generation_max_calls
        ):
            strategy.request_stop("budget_exhausted")

    def spend(self, event: BudgetEvent) -> None:
        if event.units < 0:
            raise ValueError("event units must be non-negative")
        if event.metadata.get("billable") is False:
            return
        if event.phase == "iteration":
            return
        if event.phase == "evaluation":
            remaining = self.evaluation_max_calls - self.evaluation_calls
        elif event.phase == "generation":
            remaining = self.generation_max_calls - self.generation_calls
        else:
            raise ValueError(f"unknown budget phase: {event.phase}")
        exceeds_domain = event.units > remaining
        exceeds_total = event.units > self.max_calls - self.consumed_calls
        if (exceeds_domain or exceeds_total) and not event.metadata.get("allow_overrun", False):
            raise BudgetExhaustedError(
                f"budget cannot admit {event.units} {event.phase} calls"
            )
        if event.phase == "evaluation":
            self.evaluation_calls += event.units
        else:
            self.generation_calls += event.units
        self.consumed_calls += event.units
        logger.debug("Budget spent: %s", event)

    def serialize_state(self) -> dict[str, Any]:
        remaining_calls = max(0, self.max_calls - self.consumed_calls)
        return {
            "calls": remaining_calls,
            "evaluation_calls": max(0, self.evaluation_max_calls - self.evaluation_calls),
            "generation_calls": max(0, self.generation_max_calls - self.generation_calls),
            "percentage": (remaining_calls / self.max_calls) * 100 if self.max_calls > 0 else 0,
            "max_calls": self.max_calls,
            "evaluation_max_calls": self.evaluation_max_calls,
            "generation_max_calls": self.generation_max_calls,
            "consumed_calls": self.consumed_calls,
            "evaluation_calls_consumed": self.evaluation_calls,
            "generation_calls_consumed": self.generation_calls,
        }

    @classmethod
    def restore_state(cls, state: Mapping[str, Any]):
        if "max_calls" not in state:
            raise ValueError("LMCallsBudget checkpoint is missing max_calls")
        budget = cls(
            max_calls=int(state["max_calls"]),
            evaluation_max_calls=int(
                state.get("evaluation_max_calls", state["max_calls"])
            ),
            generation_max_calls=int(
                state.get("generation_max_calls", state["max_calls"])
            ),
        )
        consumed = state.get("consumed_calls")
        if consumed is None and "calls" in state:
            consumed = budget.max_calls - int(state["calls"])
        evaluation = state.get("evaluation_calls_consumed")
        if evaluation is None and "evaluation_calls" in state:
            evaluation = budget.evaluation_max_calls - int(state["evaluation_calls"])
        generation = state.get("generation_calls_consumed")
        if generation is None and "generation_calls" in state:
            generation = budget.generation_max_calls - int(state["generation_calls"])
        budget.consumed_calls = min(budget.max_calls, max(0, int(consumed or 0)))
        budget.evaluation_calls = min(
            budget.evaluation_max_calls, max(0, int(evaluation or 0))
        )
        budget.generation_calls = min(
            budget.generation_max_calls, max(0, int(generation or 0))
        )
        return budget
