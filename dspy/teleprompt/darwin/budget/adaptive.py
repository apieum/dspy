"""Adaptive budget implementation."""

from typing import Any, Mapping
from .budget import Budget, BudgetEvent, BudgetExhaustedError, configured_limit


class AdaptiveBudget(Budget):
    """Budget that adapts allocation based on progress."""
    
    def __init__(self, total_budget: int | None = None, adaptation_factor: float = 1.2, config=None):
        self.total_budget = configured_limit(
            total_budget,
            config,
            "max_lm_calls",
            "AdaptiveBudget requires total_budget or a configuration providing max_lm_calls",
        )
        self.consumed_budget = 0
        self.adaptation_factor = adaptation_factor
        self.recent_improvements = []

    def reconcile(self, strategy) -> None:
        if self.consumed_budget >= self.total_budget:
            strategy.request_stop("budget_exhausted")

    def spend(self, event: BudgetEvent) -> None:
        if event.units < 0:
            raise ValueError("event units must be non-negative")
        if event.metadata.get("billable") is False:
            return
        if event.metadata.get("improvement") is not None:
            self.recent_improvements.append(float(event.metadata["improvement"]))
            self.recent_improvements = self.recent_improvements[-10:]
        cost = event.units
        if event.phase == "evaluation" and self.recent_improvements and sum(self.recent_improvements) > 0:
            cost = int(cost * self.adaptation_factor)
        elif event.phase == "generation":
            cost = max(2, cost)
            if self.recent_improvements and sum(self.recent_improvements) < 0.1:
                cost = int(cost * self.adaptation_factor)
        if event.phase not in {"evaluation", "generation", "iteration"}:
            raise ValueError(f"unknown budget phase: {event.phase}")
        if self.consumed_budget + cost > self.total_budget and not event.metadata.get(
            "allow_overrun", False
        ):
            raise BudgetExhaustedError("adaptive budget is exhausted")
        self.consumed_budget += cost

    def serialize_state(self) -> dict[str, Any]:
        remaining_budget = max(0, self.total_budget - self.consumed_budget)
        return {
            "budget": remaining_budget,
            "percentage": (remaining_budget / self.total_budget) * 100 if self.total_budget > 0 else 0,
            "total_budget": self.total_budget,
            "consumed_budget": self.consumed_budget,
            "adaptation_factor": self.adaptation_factor,
            "recent_improvements": list(self.recent_improvements),
        }

    @classmethod
    def restore_state(cls, state: Mapping[str, Any]):
        if "total_budget" not in state:
            raise ValueError("AdaptiveBudget checkpoint is missing total_budget")
        budget = cls(
            total_budget=int(state["total_budget"]),
            adaptation_factor=float(state.get("adaptation_factor", 1.2)),
        )
        budget.consumed_budget = min(
            budget.total_budget,
            max(0, int(state.get("consumed_budget", 0))),
        )
        budget.recent_improvements = [
            float(value) for value in state.get("recent_improvements", [])
        ][-10:]
        return budget
    
