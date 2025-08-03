"""Cohort data structure for GEPA optimization."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

import dspy

# Import Candidate with relative import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..data.candidate import Candidate
    from ..data.candidate_pool import CandidatePool


class Cohort:
    """A cohort of candidates created in a single iteration.

    Represents a group of candidates that were generated together
    and share the same iteration context. The iteration_id is stored
    in the candidates themselves as they're always from the same iteration.
    """

    def __init__(self, *candidates: 'Candidate', creation_timestamp: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize cohort with candidates.

        Args:
            *candidates: Variable number of candidate arguments, or a single list of candidates
            creation_timestamp: When cohort was created (defaults to current time)
            metadata: Optional metadata dict
        """
        # Handle both Cohort(candidate1, candidate2) and Cohort([candidate1, candidate2])
        if len(candidates) == 1 and isinstance(candidates[0], list):
            self.candidates: List['Candidate'] = candidates[0]
        else:
            self.candidates: List['Candidate'] = list(candidates)

        self.creation_timestamp: float = creation_timestamp or time.time()
        self.metadata: Dict[str, Any] = metadata or {}

    @property
    def iteration_id(self) -> int:
        """Get iteration ID from candidates (they're all from same iteration)."""
        if not self.candidates:
            return -1
        return self.candidates[0].generation_number

    def is_empty(self) -> bool:
        """Check if cohort has no candidates."""
        return len(self.candidates) == 0

    def size(self) -> int:
        """Number of candidates in this cohort."""
        return len(self.candidates)

    def add_candidate(self, candidate: 'Candidate') -> None:
        """Add a candidate to this cohort."""
        self.candidates.append(candidate)

    def update_scores(self, pool:'CandidatePool') -> None:
        """Update the candidate pool scores with the best candidate for each task in this cohort.

        The cohort analyzes its candidates, finds the best performer for each task,
        and updates the score matrix with individual task updates.

        Args:
            pool: CandidatePool to update scores with best candidates per task
        """
        if not self.candidates:
            return

        # Determine the number of tasks from the first candidate
        if not self.candidates[0].task_scores:
            return

        num_tasks = len(self.candidates[0].task_scores)

        # For each task, find the best candidate and update the matrix
        for task_id in range(num_tasks):
            best_candidate = self.candidates[0]

            # Find the candidate with the highest score for this task
            for candidate in self.candidates:
                best_candidate = candidate.best_for_task(task_id, best_candidate)

            pool.update_score(task_id, best_candidate)


class FilteredCohort(Cohort):
    """Cohort after evaluation filtering with scores compiled."""

    def __init__(self, *candidates: 'Candidate', filtered_count: int = 0, evaluation_summary: str = "", **kwargs):
        super().__init__(*candidates, **kwargs)
        self.filtered_count = filtered_count
        self.evaluation_summary = evaluation_summary
