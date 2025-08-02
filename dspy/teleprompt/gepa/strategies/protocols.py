"""Protocol interfaces for GEPA optimization components.

These protocols define the contracts for GEPA optimization,
allowing researchers to experiment with different algorithms for each step.
"""

from abc import abstractmethod
from typing import Callable, List, Protocol

import dspy


# Forward declarations for type hints
class GenerationData:
    """Forward declaration - will be defined in generation structures."""
    pass

class Candidate: 
    """Forward declaration - will be defined in data structures."""
    pass

class CandidatePool:
    """Forward declaration - will be defined in data structures."""
    pass

class ScoreMatrix:
    """Forward declaration - will be defined in data structures."""
    pass


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


class Scoring(Protocol):
    """Protocol for calculating candidate performance scores.
    
    This strategy owns the metric and is responsible for evaluating
    candidates to produce scores used by other strategies.
    """
    
    @abstractmethod
    def calculate_scores(self, candidates: List[Candidate], 
                        data: List[dspy.Example],
                        candidate_pool: "CandidatePool") -> "ScoreMatrix":
        """Calculate performance scores for candidates and update candidate pool.
        
        Args:
            candidates: List of candidates to score
            data: Evaluation data (e.g., Pareto set)
            candidate_pool: Pool to update with scored candidates
            
        Returns:
            ScoreMatrix for compatibility with filtering strategies
        """
        ...
    
    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the metric function used by this strategy."""
        ...


class Selection(Protocol):
    """Protocol for filtering candidates based on performance scores.
    
    This strategy uses scores (not metric directly) to decide which
    candidates should continue to the next generation.
    """
    
    @abstractmethod
    def filter(self, candidate_pool: "CandidatePool", score_matrix: "ScoreMatrix") -> List[Candidate]:
        """Filter candidates based on performance scores.
        
        Args:
            candidate_pool: Pool containing all candidates and their scores
            score_matrix: Matrix providing access to scoring data
            
        Returns:
            List of selected (surviving) candidates
        """
        ...


class Generator(Protocol):
    """Protocol for generating new candidates from existing pool.
    
    This strategy implements the genetic operations (mutation, crossover, etc)
    to create new candidate generations.
    """
    
    @abstractmethod
    def generate(self, parent_candidates: List[Candidate],
                feedback_data: List[dspy.Example], 
                iteration: int) -> "GenerationData":
        """Generate new candidates from parent candidates.
        
        Args:
            parent_candidates: Selected parent candidates for generation
            feedback_data: Data for generating feedback/mutations
            iteration: Current iteration number
            
        Returns:
            GenerationData containing newly generated candidates
        """
        ...


class Evaluator(Protocol):
    """Protocol for evaluating and filtering new candidates.
    
    This strategy owns a metric and decides which newly generated
    candidates should be promoted (kept) vs discarded.
    """
    
    @abstractmethod
    def evaluate(self, generation: "GenerationData",
                feedback_data: List[dspy.Example]) -> "GenerationData":
        """Evaluate new candidates and filter based on promotion criteria.
        
        Args:
            generation: Newly generated candidates to evaluate
            feedback_data: Data for evaluation
            
        Returns:
            GenerationData containing only promoted (worthy) candidates
        """
        ...
    
    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the metric function used by this strategy."""
        ...