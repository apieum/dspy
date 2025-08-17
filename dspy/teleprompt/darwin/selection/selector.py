"""Selection protocol for GEPA optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING, List
from typing_extensions import Optional
import dspy

if TYPE_CHECKING:
    from ..data import Candidate, Survivors, Parents
    from ..budget import Budget


class Selector:
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
