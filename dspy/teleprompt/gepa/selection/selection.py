"""Selection protocol for GEPA optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data import Candidate, CandidatePool, Cohort
    from ..budget import Budget


class Selection(CompilationObserver):
    """Protocol for filtering candidates based on performance data.

    This component uses scores and candidate data to decide which
    candidates should continue to the next generation.
    """

    @abstractmethod
    def filter(self, pool:"CandidatePool", budget:"Budget"=None) -> "Cohort":
        """Filter candidates strategy, called directly in GEPA core.

        Args:
            pool: Candidate pool to filter
            budget: Optional budget parameter for tracking selection costs

        Returns:
            Cohort of selected (surviving) candidates
        """
        ...
