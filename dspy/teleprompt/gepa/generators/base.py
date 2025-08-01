"""Base interfaces for candidate generation strategies."""

from abc import ABC, abstractmethod
from typing import Iterable, List

import dspy

from ..data.candidate_pool import CandidatePool


class CandidateGenerator(ABC):
    """Interface for candidate generation strategies.
    
    Handles the generation of new candidates from the current pool using
    various strategies like mutation, crossover, or other evolutionary operators.
    Follows the paper's approach of using generic feedback data for guidance.
    """
    
    @abstractmethod
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[dspy.Example], 
                          iteration: int) -> List[dspy.Module]:
        """Generate new candidates from the current pool.
        
        Args:
            candidate_pool: Current pool of candidates with scores and lineages
            feedback_data: Generic feedback examples for mutation guidance (paper's feedback set)
            iteration: Current iteration number for timing decisions
            
        Returns:
            List of new candidates to be evaluated and potentially added to pool
        """
        raise NotImplementedError