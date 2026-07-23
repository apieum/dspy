"""Generic selection protocol for Darwin algorithms."""

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing_extensions import Optional
from ..observers import Channel

if TYPE_CHECKING:
    from ..data import Candidate, Cohort
    from ..budget import Budget
    from ..config import DarwinConfig


class Selector(Channel):
    """Protocol for promoting and selecting candidate cohorts."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of candidates tracked by this selector."""
        ...

    @abstractmethod
    def promote(self, survivors: "Cohort", budget: Optional['Budget'] = None) -> "Cohort":
        """Promote candidates into the next algorithm-defined cohort.

        Args:
            survivors: Cohort accepted by the preceding phase

        Returns:
            Cohort ready for the next phase
        """
        ...
    @abstractmethod
    def best_candidate(self) -> "Candidate":
        """Return the best candidate.

        Returns:
            The best candidate.
        """
        ...
