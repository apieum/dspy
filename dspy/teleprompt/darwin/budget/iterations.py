"""Iterations budget implementation."""

from typing import Any, Mapping
from .budget import Budget, BudgetEvent, BudgetExhaustedError, configured_limit


class IterationBudget(Budget):
    """Budget that limits by number of iterations."""
    
    def __init__(self, max_iterations: int | None = None, config=None):
        self.max_iterations = configured_limit(
            max_iterations,
            config,
            "max_iterations",
            "IterationBudget requires max_iterations or a configuration providing it",
        )
        self.current_iteration = 0

    def reconcile(self, strategy) -> None:
        if self.current_iteration >= self.max_iterations:
            strategy.request_stop("budget_exhausted")

    def spend(self, event: BudgetEvent) -> None:
        if event.units < 0:
            raise ValueError("event units must be non-negative")
        if event.phase != "iteration":
            return
        if self.current_iteration + event.units > self.max_iterations:
            raise BudgetExhaustedError("iteration budget is exhausted")
        self.current_iteration += event.units

    def serialize_state(self) -> dict[str, Any]:
        return {
            "iterations": max(0, self.max_iterations - self.current_iteration),
            "percentage": (
                max(0, self.max_iterations - self.current_iteration)
                / self.max_iterations
                * 100
                if self.max_iterations > 0
                else 0
            ),
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
        }

    @classmethod
    def restore_state(cls, state: Mapping[str, Any]):
        if "max_iterations" not in state:
            raise ValueError("IterationBudget checkpoint is missing max_iterations")
        budget = cls(max_iterations=int(state["max_iterations"]))
        budget.current_iteration = min(
            budget.max_iterations,
            max(0, int(state.get("current_iteration", 0))),
        )
        return budget
