"""LLM calls budget implementation."""

from .budget import Budget


class LLMCallsBudget(Budget):
    """Budget that tracks LLM API calls."""
    
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.consumed_calls = 0
        
    def can_afford(self, cost: int, operation_type: str) -> bool:
        return self.consumed_calls + cost <= self.max_calls
        
    def consume(self, cost: int, operation_type: str) -> None:
        self.consumed_calls += cost
        
    def is_empty(self) -> bool:
        return self.consumed_calls >= self.max_calls
        
    def get_remaining(self) -> int:
        return max(0, self.max_calls - self.consumed_calls)