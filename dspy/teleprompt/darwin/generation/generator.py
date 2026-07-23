"""Generic generation protocol for Darwin algorithms."""

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional
import dspy
from ..observers import Channel

if TYPE_CHECKING:
    from ..data.cohort import Cohort
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
    def generate(self, parents: "Cohort", budget=None) -> "Cohort":
        """Generate new candidates from parent candidates.

        Args:
            parents: Input cohort for generation
            budget: Optional budget parameter for tracking generation costs

        Returns:
            A cohort containing generated candidates
        """
        ...

    def generate_batch(
        self,
        parents: "Cohort",
        count: int,
        budget=None,
        *,
        sampling_strategy=None,
        batch_sampler=None,
        rng=None,
    ) -> "Cohort":
        """Generate a batch of independent proposals from the same parents.

        The concrete ``generate`` implementation chooses the output cohort
        type. The shared method only combines those outputs and never assumes
        a lifecycle role such as ``NewBorns``.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        parent_tasks = (
            sampling_strategy.sample(parents, count, rng=rng)
            if sampling_strategy is not None
            else [parents] * count
        )
        feedback_batches = (
            # The sampler must see the complete reflection pool.  ``feedback_data``
            # is retained as the direct-generator fallback, but strategies should
            # provide ``feedback_pool`` so successive GEPA generations do not keep
            # recycling the same initial minibatch.
            batch_sampler.sample(
                getattr(self, "feedback_pool", getattr(self, "feedback_data", [])),
                len(parent_tasks),
                rng=rng,
            )
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
        generated_cohorts = []
        for task in tasks:
            generated_cohorts.append(self.generate_task(task, budget))
        output_type = type(generated_cohorts[0]) if generated_cohorts else type(parents)
        proposals = [
            candidate
            for generated in generated_cohorts
            for candidate in generated
        ]
        return output_type(*proposals, iteration=parents.iteration)

    def generate_task(self, task, budget=None) -> "Cohort":
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
