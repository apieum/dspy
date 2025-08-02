"""Adaptive budget implementation."""

from .budget import Budget


class AdaptiveBudget(Budget):
    """Budget that adapts allocation based on progress."""
    
    def __init__(self, total_budget: int, adaptation_factor: float = 1.2):
        self.total_budget = total_budget
        self.consumed_budget = 0
        self.adaptation_factor = adaptation_factor
        self.recent_improvements = []
        
    def can_afford(self, cost: int, operation_type: str) -> bool:
        return self.consumed_budget + cost <= self.total_budget
        
    def consume(self, cost: int, operation_type: str) -> None:
        self.consumed_budget += cost
        
    def is_empty(self) -> bool:
        return self.consumed_budget >= self.total_budget
        
    def get_remaining(self) -> int:
        return max(0, self.total_budget - self.consumed_budget)
        
    def record_improvement(self, improvement: float) -> None:
        """Record performance improvement for adaptive allocation."""
        self.recent_improvements.append(improvement)
        # Keep only recent history
        if len(self.recent_improvements) > 10:
            self.recent_improvements = self.recent_improvements[-10:]