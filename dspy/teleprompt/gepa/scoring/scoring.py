"""Scoring protocol for GEPA optimization."""

from abc import abstractmethod
from typing import Callable, List, Protocol, TYPE_CHECKING
import dspy
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data.candidate import Candidate
    from ..data.candidate_pool import CandidatePool  
    from ..data.score_matrix import ScoreMatrix


class Scoring(CompilationObserver):
    """Protocol for calculating candidate performance scores.
    
    This component owns the metric and is responsible for evaluating
    candidates to produce scores used by other components.
    """
    
    @abstractmethod
    def calculate_scores(self, candidates: List["Candidate"], 
                        data: List[dspy.Example],
                        candidate_pool: "CandidatePool") -> "ScoreMatrix":
        """Calculate performance scores for candidates and update candidate pool.
        
        Args:
            candidates: List of candidates to score
            data: Evaluation data (e.g., Pareto set)
            candidate_pool: Pool to update with scored candidates
            
        Returns:
            ScoreMatrix for compatibility with filtering
        """
        ...
    
    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the metric function used by this component."""
        ...