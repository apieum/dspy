"""Generator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional
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

    def generate_batch(self, parents: "Parents", count: int, budget=None) -> "NewBorns":
        """Generate a batch of independent proposals from the same parents.

        GEPA evaluates proposal batches before promotion. Keeping batching in
        the shared generator interface lets other Darwin phases, including a
        future MIPRO phase, feed candidates into the same evaluator and
        selector pipeline.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        from ..data.cohort import NewBorns

        proposals = []
        for _ in range(count):
            proposals.extend(self.generate(parents, budget).to_list())
        return NewBorns(*proposals, iteration=parents.iteration)

    def start_compilation(
        self,
        student: dspy.Module,
        dataset_manager=None,
        *,
        feedback_data: Optional[List[dspy.Example]] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize generator state for a compilation run."""
        self.dataset_manager = dataset_manager
        self.verbose = verbose
