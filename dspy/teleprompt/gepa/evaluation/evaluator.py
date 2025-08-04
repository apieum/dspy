"""Evaluator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import Callable, List, Protocol, TYPE_CHECKING
import dspy
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data.cohort import Cohort
    from ..data.candidate_pool import CandidatePool


class Evaluator(CompilationObserver):
    """Protocol for evaluating and filtering new candidates.
    
    This component owns a metric and decides which newly generated
    candidates should be promoted (kept) vs discarded.
    """
    
    @abstractmethod
    def evaluate(self, cohort: "Cohort") -> "Cohort":
        """Evaluate new candidates and filter based on promotion criteria.
        
        Args:
            cohort: Newly generated candidates to evaluate
            
        Returns:
            Cohort containing only promoted (worthy) candidates
        """
        ...
    
    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the metric function used by this component."""
        ...
    
    
