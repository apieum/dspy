"""Budget protocol for GEPA optimization."""

from abc import abstractmethod
from typing import Dict, Any, Optional
from ..compilation_observer import CompilationObserver
import dspy


class Budget(CompilationObserver):
    """Protocol for managing optimization budget with flexible cost tracking."""
    
    def __float__(self) -> float:
        """Convert budget to float (remaining budget value)."""
        remaining = self.get_remaining()
        if isinstance(remaining, dict):
            # Get the primary budget value (first key)
            primary_key = next(iter(remaining.keys()))
            return float(remaining[primary_key])
        return float(remaining)
    
    def __int__(self) -> int:
        """Convert budget to int (remaining budget value)."""
        remaining = self.get_remaining()
        if isinstance(remaining, dict):
            # Get the primary budget value (first key)
            primary_key = next(iter(remaining.keys()))
            return int(remaining[primary_key])
        return int(remaining)
    
    def __gt__(self, other) -> bool:
        """Magic comparison for `budget > value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) > other
        return NotImplemented
    
    def __lt__(self, other) -> bool:
        """Magic comparison for `budget < value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) < other
        return NotImplemented
    
    def __le__(self, other) -> bool:
        """Magic comparison for `budget <= value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) <= other
        return NotImplemented
    
    def __ge__(self, other) -> bool:
        """Magic comparison for `budget >= value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) >= other
        return NotImplemented
    
    def __eq__(self, other) -> bool:
        """Magic comparison for `budget == value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) == other
        return NotImplemented
    
    def __ne__(self, other) -> bool:
        """Magic comparison for `budget != value`."""
        return not self.__eq__(other)
    
    def spend_on_evaluation(self, module: dspy.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of evaluating a candidate module (LLM calls).
        
        Args:
            module: DSPy module that was evaluated
            metadata: Optional details like {"phase": "minibatch", "examples": 3}
        """
        pass  # Override in child classes if needed
        
    def spend_on_generation(self, module: Optional[dspy.Module] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of generating new candidates (LLM calls for reflection/mutation).
        
        Args:
            module: Optional module being mutated/generated from
            metadata: Optional details like {"type": "reflection", "strategy": "mutation"}
        """
        pass  # Override in child classes if needed
        
    def spend_on_selection(self, candidates_in_pool: int, candidates_selected: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of candidate selection (usually algorithmic, no LLM calls).
        
        Args:
            candidates_in_pool: Total candidates available for selection
            candidates_selected: Number of candidates actually selected
            metadata: Optional details like {"strategy": "pareto", "tasks": 150}
        """
        pass  # Override in child classes if needed
    
    @abstractmethod
    def get_remaining(self) -> dict:
        """Get remaining budget breakdown."""
        ...