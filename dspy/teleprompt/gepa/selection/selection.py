"""Selection protocol for GEPA optimization."""

from abc import abstractmethod
from typing import List, Dict, Optional, TYPE_CHECKING
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data import Candidate, CandidatePool
    from ..budget import Budget


class Selection(CompilationObserver):
    """Protocol for filtering candidates based on performance data.

    This component uses scores and candidate data to decide which
    candidates should continue to the next generation.
    """

    @abstractmethod
    def filter(self, pool:"CandidatePool", budget:"Budget"=None):
        """
        Filter candidates strategy, called directly in GEPA core.

        Args:
            pool: Candidate pool to filter
            budget: Optional budget parameter for tracking selection costs

        Returns:
            List of selected (surviving) candidates
        """
        ...

    @abstractmethod
    def filter_candidates(self, candidates: List["Candidate"]) -> List["Candidate"]:
        """Filter candidates based on candidate objects directly.

        Args:
            candidates: List of candidate objects to filter

        Returns:
            List of selected (surviving) candidates
        """
        ...

    @abstractmethod
    def filter_scores(self, task_scores: Dict[int, "Candidate"]) -> List["Candidate"]:
        """Filter candidates based on task score data.

        Args:
            task_scores: Dict with task_id -> best_candidate mappings

        Returns:
            List of selected (surviving) candidates
        """
        ...

    @abstractmethod
    def filter_generation(self, gen_id: int, task_scores: Dict[int, "Candidate"]) -> List["Candidate"]:
        """Filter candidates from a specific generation.

        Args:
            gen_id: Generation ID to filter
            task_scores: Task score data for that generation

        Returns:
            List of selected candidates from that generation
        """
        ...

    @abstractmethod
    def filter_generation_history(self, gen_id: int, update_index: int, task_scores: Dict[int, "Candidate"]) -> List["Candidate"]:
        """Filter candidates from a specific historical snapshot.

        Args:
            gen_id: Generation ID
            update_index: Index of the historical snapshot within that generation
            task_scores: Historical task score data

        Returns:
            List of selected candidates from that historical snapshot
        """
        ...
