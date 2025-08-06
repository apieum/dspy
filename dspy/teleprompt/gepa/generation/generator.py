"""Generator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data.cohort import Parents, NewBorns


class Generator(CompilationObserver):
    """Protocol for generating new candidates from parents.

    This component implements the genetic operations (mutation, crossover, etc)
    to create new candidate generations.
    """

    @abstractmethod
    def generate(self, parents: "Parents", budget=None) -> "NewBorns":
        """Generate new candidates from parent candidates.

        Args:
            parents: Parents cohort of parent candidates for generation
            budget: Optional budget parameter for tracking generation costs

        Returns:
            NewBorns cohort containing newly generated candidates
        """
        ...
