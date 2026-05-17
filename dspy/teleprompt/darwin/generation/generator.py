"""Generator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING, List
import dspy
from ..observers import Channel

if TYPE_CHECKING:
    from ..data.cohort import Parents, NewBorns
    from ..config import DarwinConfig


class Generator(Channel):
    """Protocol for generating new candidates from parents.

    This component implements the genetic operations (mutation, merge, etc)
    to create new candidate generations.
    """

    def __init__(self):
        """Initialize generator with observer support."""
        super().__init__()

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

    # Lifecycle methods removed - strategy owns component lifecycle
    # Components are called directly by strategy with specific data
