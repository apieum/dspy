"""Evaluator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import Callable, List, Protocol, TYPE_CHECKING
import dspy

if TYPE_CHECKING:
    from ..generation.generation import Generation


class Evaluator(Protocol):
    """Protocol for evaluating and filtering new candidates.
    
    This component owns a metric and decides which newly generated
    candidates should be promoted (kept) vs discarded.
    """
    
    @abstractmethod
    def evaluate(self, generation: "Generation",
                evaluation_data: List[dspy.Example]) -> "Generation":
        """Evaluate new candidates and filter based on promotion criteria.
        
        Args:
            generation: Newly generated candidates to evaluate
            evaluation_data: Data for evaluation
            
        Returns:
            Generation containing only promoted (worthy) candidates
        """
        ...
    
    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the metric function used by this component."""
        ...