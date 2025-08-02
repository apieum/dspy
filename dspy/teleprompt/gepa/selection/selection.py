"""Selection protocol for GEPA optimization."""

from abc import abstractmethod
from typing import List, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.candidate import Candidate
    from ..data.candidate_pool import CandidatePool  
    from ..data.score_matrix import ScoreMatrix


class Selection(Protocol):
    """Protocol for filtering candidates based on performance scores.
    
    This component uses scores (not metric directly) to decide which
    candidates should continue to the next generation.
    """
    
    @abstractmethod
    def filter(self, candidate_pool: "CandidatePool", score_matrix: "ScoreMatrix") -> List["Candidate"]:
        """Filter candidates based on performance scores.
        
        Args:
            candidate_pool: Pool containing all candidates and their scores
            score_matrix: Matrix providing access to scoring data
            
        Returns:
            List of selected (surviving) candidates
        """
        ...