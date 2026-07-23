"""Iterations budget implementation."""

from typing import Dict, Any, Mapping, Optional
import dspy
from .budget import Budget


class IterationBudget(Budget):
    """Budget that limits by number of iterations."""
    
    def __init__(self, max_iterations: Optional[int] = None, config=None):
        if config is not None:
            max_iterations = config.max_iterations
        if max_iterations is None:
            raise TypeError("IterationBudget requires a configuration or max_iterations")
        self.config = config
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def reset(self) -> None:
        self.current_iteration = 0
        
        
    def _get_remaining(self) -> dict:
        remaining_iterations = max(0, self.max_iterations - self.current_iteration)
        return {
            "iterations": remaining_iterations,
            "percentage": (remaining_iterations / self.max_iterations) * 100 if self.max_iterations > 0 else 0
        }

    def is_exhausted(self) -> bool:
        return self.current_iteration >= self.max_iterations

    def can_spend(self, phase: str, units: int = 1) -> bool:
        del phase, units
        return not self.is_exhausted()

    def serialize_state(self) -> dict[str, Any]:
        return {
            **self._get_remaining(),
            "current_iteration": self.current_iteration,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state:
            self.current_iteration = min(
                self.max_iterations,
                max(0, int(state.get("current_iteration", 0))),
            )
    
    
    def finish_iteration(self, iteration: int, cohort=None) -> None:
        """Count completed iterations rather than individual phases."""
        del cohort
        self.current_iteration = iteration
