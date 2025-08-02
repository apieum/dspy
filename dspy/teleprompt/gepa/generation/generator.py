"""Generator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import List, Protocol, TYPE_CHECKING
import dspy

if TYPE_CHECKING:
    from ..data.candidate import Candidate
    from .generation import Generation


class Generator(Protocol):
    """Protocol for generating new candidates from existing pool.
    
    This component implements the genetic operations (mutation, crossover, etc)
    to create new candidate generations.
    """
    
    @abstractmethod
    def generate(self, parent_candidates: List["Candidate"],
                feedback_data: List[dspy.Example], 
                iteration: int) -> "Generation":
        """Generate new candidates from parent candidates.
        
        Args:
            parent_candidates: Selected parent candidates for generation
            feedback_data: Data for generating feedback/mutations
            iteration: Current iteration number
            
        Returns:
            Generation containing newly generated candidates
        """
        ...