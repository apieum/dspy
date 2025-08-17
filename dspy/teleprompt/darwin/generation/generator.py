"""Generator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING, List
import dspy

if TYPE_CHECKING:
    from ..data.cohort import Parents, NewBorns


class Generator:
    """Protocol for generating new candidates from parents.

    This component implements the genetic operations (mutation, merge, etc)
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

    # Lifecycle methods (no-op implementations by default)
    def start_compilation(self, student: dspy.Module, split_strategy=None, verbose: bool = False) -> None:
        """Called when compilation begins. Components can prepare resources."""
        pass

    def finish_compilation(self, result: dspy.Module) -> None:
        """Called when compilation ends. Components can cleanup/log results."""
        pass

    def start_iteration(self, iteration: int, cohort, budget) -> None:
        """Called at start of each optimization iteration."""
        pass

    def finish_iteration(self, iteration: int, filtered_cohort, budget) -> None:
        """Called after each optimization iteration completes."""
        pass
