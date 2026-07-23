"""Adaptive budget implementation."""

from typing import Dict, Any, Optional
import dspy
from .budget import Budget


class AdaptiveBudget(Budget):
    """Budget that adapts allocation based on progress."""
    
    def __init__(self, total_budget: Optional[int] = None, adaptation_factor: float = 1.2, config=None):
        if config is not None:
            total_budget = config.max_lm_calls
        if total_budget is None:
            raise TypeError("AdaptiveBudget requires a configuration or total_budget")
        self.config = config
        self.total_budget = total_budget
        self.consumed_budget = 0
        self.adaptation_factor = adaptation_factor
        self.recent_improvements = []

    def reset(self) -> None:
        self.consumed_budget = 0
        self.recent_improvements.clear()
        
        
    def spend_on_evaluation(self, module: dspy.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of evaluating a candidate module - adaptive cost based on recent progress."""
        base_cost = 1
        # Adapt cost based on recent improvements
        if self.recent_improvements and sum(self.recent_improvements) > 0:
            base_cost = int(base_cost * self.adaptation_factor)
        self.consumed_budget = min(self.total_budget, self.consumed_budget + base_cost)
        
    def spend_on_generation(self, module: Optional[dspy.Module] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of generating new candidates - higher cost if improvements are low."""
        base_cost = 2  # Generation typically more expensive
        if self.recent_improvements and sum(self.recent_improvements) < 0.1:
            base_cost = int(base_cost * self.adaptation_factor)
        self.consumed_budget = min(self.total_budget, self.consumed_budget + base_cost)
        
    def get_remaining(self) -> dict:
        remaining_budget = max(0, self.total_budget - self.consumed_budget)
        return {
            "budget": remaining_budget,
            "percentage": (remaining_budget / self.total_budget) * 100 if self.total_budget > 0 else 0
        }

    def is_exhausted(self) -> bool:
        return self.consumed_budget >= self.total_budget

    def can_spend(self, phase: str, units: int = 1) -> bool:
        del phase
        if units < 0:
            raise ValueError("units must be non-negative")
        return units <= self.total_budget - self.consumed_budget

    def serialize_state(self) -> dict[str, Any]:
        return {
            **self.get_remaining(),
            "consumed_budget": self.consumed_budget,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if state:
            self.consumed_budget = min(
                self.total_budget,
                max(0, int(state.get("consumed_budget", 0))),
            )
        
    def record_improvement(self, improvement: float) -> None:
        """Record performance improvement for adaptive allocation."""
        self.recent_improvements.append(improvement)
        # Keep only recent history
        if len(self.recent_improvements) > 10:
            self.recent_improvements = self.recent_improvements[-10:]
    
