"""Abstract budget tracker interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BudgetTracker(ABC):
    """Abstract budget tracker interface."""
    
    @abstractmethod
    def add_minibatch_cost(self, cost: int):
        """Add cost for minibatch evaluations."""
        pass

    @abstractmethod
    def add_reflection_cost(self, cost: int = 1):
        """Add cost for reflection/mutation operations."""
        pass

    @abstractmethod
    def add_pareto_cost(self, cost: int):
        """Add cost for Pareto evaluations."""
        pass
    
    @abstractmethod
    def add_iteration_cost(self, cost: int = 1):
        """Add cost for iteration overhead."""
        pass

    @abstractmethod
    def warn_iteration_start(self, iteration: int):
        """Notify budget tracker that new iteration is starting."""
        pass

    @abstractmethod
    def has_budget(self) -> bool:
        """Check if budget allows continuing."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Get budget usage statistics."""
        pass