"""Selection protocol for GEPA optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing_extensions import Callable, Optional
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data import Candidate, Cohort, Survivors, Parents
    from ..budget import Budget


class Selector(CompilationObserver):
    """Protocol for filtering candidates based on performance data.

    This component uses scores and candidate data to decide which
    candidates should continue to the next generation.
    """
    def size(self) -> int:
        """Return the size of the selector.

        Returns:
            The size of the selector.
        """
        return len(self.task_wins)

    @abstractmethod
    def filter(self, survivors: "Survivors", budget:Optional['Budget']=None) -> "Survivors":
        """Filter candidates strategy, called directly in GEPA core.

        Args:
            survivors: Current cohort of survivors to filter from
            budget: Optional budget parameter for tracking selection costs

        Returns:
            Survivors cohort of selected candidates
        """
        ...
    @abstractmethod
    def promote(self, survivors: "Survivors", budget: Optional['Budget'] = None) -> "Parents":
        """Promote candidates strategy, called directly in GEPA core.

        Args:
            survivors: Survivors cohort to promote to parents

        Returns:
            Parents cohort ready for reproduction
        """
        ...
    @abstractmethod
    def best_candidate(self) -> "Candidate":
        """Return the best candidate.

        Returns:
            The best candidate.
        """
        ...
