"""Generator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import List, Protocol, TYPE_CHECKING
import dspy
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data.candidate import Candidate
    from ..data.cohort import Cohort


class Generator(CompilationObserver):
    """Protocol for generating new candidates from existing pool.
    
    This component implements the genetic operations (mutation, crossover, etc)
    to create new candidate generations.
    """
    
    @abstractmethod
    def generate(self, parent_candidates: List["Candidate"], iteration: int) -> "Cohort":
        """Generate new candidates from parent candidates.
        
        Args:
            parent_candidates: Selected parent candidates for generation
            iteration: Current iteration number
            
        Returns:
            Cohort containing newly generated candidates
        """
        ...
    
    @abstractmethod
    def generate_from_parents(self, parent_candidates: List["Candidate"]) -> "Cohort":
        """Generate new candidates from parent candidates (simplified interface).
        
        Used by ParetoFrontier.generate() method for dependency injection pattern.
        
        Args:
            parent_candidates: Selected parent candidates for generation
            
        Returns:
            Cohort containing newly generated candidates (created by this generator)
        """
        ...
    
    @abstractmethod  
    def create_empty_cohort(self) -> "Cohort":
        """Create an empty cohort of the type this generator produces.
        
        Used when no parent candidates are available.
        
        Returns:
            Empty cohort of the appropriate type
        """
        ...
    
