"""Adaptive budget implementation."""

from typing import Dict, Any, Optional
import dspy
from .budget import Budget


class AdaptiveBudget(Budget):
    """Budget that adapts allocation based on progress."""
    
    def __init__(self, total_budget: int, adaptation_factor: float = 1.2):
        self.total_budget = total_budget
        self.consumed_budget = 0
        self.adaptation_factor = adaptation_factor
        self.recent_improvements = []
        
        
    def spend_on_evaluation(self, module: dspy.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of evaluating a candidate module - adaptive cost based on recent progress."""
        base_cost = 1
        # Adapt cost based on recent improvements
        if self.recent_improvements and sum(self.recent_improvements) > 0:
            base_cost = int(base_cost * self.adaptation_factor)
        self.consumed_budget += base_cost
        
    def spend_on_generation(self, module: Optional[dspy.Module] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of generating new candidates - higher cost if improvements are low."""
        base_cost = 2  # Generation typically more expensive
        if self.recent_improvements and sum(self.recent_improvements) < 0.1:
            base_cost = int(base_cost * self.adaptation_factor)
        self.consumed_budget += base_cost
        
    def get_remaining(self) -> dict:
        remaining_budget = max(0, self.total_budget - self.consumed_budget)
        return {
            "budget": remaining_budget,
            "percentage": (remaining_budget / self.total_budget) * 100 if self.total_budget > 0 else 0
        }
        
    def record_improvement(self, improvement: float) -> None:
        """Record performance improvement for adaptive allocation."""
        self.recent_improvements.append(improvement)
        # Keep only recent history
        if len(self.recent_improvements) > 10:
            self.recent_improvements = self.recent_improvements[-10:]
    
