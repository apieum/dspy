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

    def generate_batch(
        self,
        parents: "Parents",
        count: int,
        budget=None,
        *,
        sampling_strategy=None,
        batch_sampler=None,
        rng=None,
    ) -> "NewBorns":
        """Generate a batch of independent proposals from the same parents.

        GEPA evaluates proposal batches before promotion. Keeping batching in
        the shared generator interface lets other Darwin phases, including a
        future MIPRO phase, feed candidates into the same evaluator and
        selector pipeline.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        from ..data.cohort import NewBorns

        parent_tasks = (
            sampling_strategy.sample(parents, count, rng=rng)
            if sampling_strategy is not None
            else [parents] * count
        )
        feedback_batches = (
            batch_sampler.sample(getattr(self, "feedback_data", []), len(parent_tasks), rng=rng)
            if batch_sampler is not None
            else [None] * len(parent_tasks)
        )
        from .sampling import ProposalTask
        tasks = [
            ProposalTask(task_parents, list(feedback_data or []), {"proposal_index": index})
            for index, (task_parents, feedback_data) in enumerate(
                zip(parent_tasks, feedback_batches, strict=True)
            )
        ]
        proposals = []
        for task in tasks:
            proposals.extend(self.generate_task(task, budget).to_list())
        return NewBorns(*proposals, iteration=parents.iteration)

    def generate_task(self, task, budget=None) -> "NewBorns":
        """Generate one explicit proposal task while preserving ``generate``."""
        try:
            self._active_proposal_task = task
            self._active_feedback_data = task.feedback_data
            return self.generate(task.parents, budget)
        finally:
            self._active_proposal_task = None
            self._active_feedback_data = None

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
