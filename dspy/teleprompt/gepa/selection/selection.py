"""Selection protocol for GEPA optimization."""

from abc import abstractmethod
from typing import List, Protocol, TYPE_CHECKING
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data.candidate import Candidate
    from ..data.candidate_pool import CandidatePool  
    from ..data.score_matrix import ScoreMatrix


class Selection(CompilationObserver):
    """Protocol for filtering candidates based on performance scores.
    
    This component uses scores (not metric directly) to decide which
    candidates should continue to the next generation.
    """
    
    @abstractmethod
    def filter(self, task_score_data: dict) -> List["Candidate"]:
        """Filter candidates based on task score data provided by ScoreMatrix.
        
        The ScoreMatrix decides what data to provide and calls this method.
        Selector should NEVER access candidate_pool or score_matrix directly.
        
        Args:
            task_score_data: Dict with task_id -> List[(candidate, score)] mappings
                            prepared by ScoreMatrix for this selection strategy
            
        Returns:
            List of selected (surviving) candidates
        """
        ...