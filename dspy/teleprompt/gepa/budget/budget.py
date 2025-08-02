"""Budget protocol for GEPA optimization."""

from abc import abstractmethod
from typing import Protocol


class Budget(Protocol):
    """Protocol for managing optimization budget (LLM calls, iterations, etc)."""
    
    @abstractmethod
    def is_empty(self) -> bool:
        """Check if budget is exhausted."""
        ...
    
    @abstractmethod
    def consume(self, cost: int, cost_type: str = "llm_call") -> None:
        """Consume budget for an operation."""
        ...
    
    @abstractmethod
    def can_afford(self, cost: int, cost_type: str = "llm_call") -> bool:
        """Check if budget can afford an operation."""
        ...
    
    @abstractmethod
    def get_remaining(self) -> dict:
        """Get remaining budget breakdown."""
        ...