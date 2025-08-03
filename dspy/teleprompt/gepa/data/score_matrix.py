"""ScoreMatrix for tracking best candidates per task."""

from typing import Dict, List, Optional, Any

# Import Candidate and Cohort with relative import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .candidate import Candidate
    from .cohort import Cohort


class ScoreMatrix:
    """Task-indexed matrix storing best candidates per task.

    - task_scores: List where position corresponds to task_id, value is index into candidates list

    This allows efficient memory usage and tracking of candidates that excel in multiple tasks.
    """

    def __init__(self):
        # task_id -> best candidate for that task
        self.task_scores: Dict[int, 'Candidate'] = {}

    def update_score(self, task_id: int, candidate: 'Candidate') -> None:
        """Update the score matrix with a single candidate for a specific task.

        The cohort determines the best candidate per task and calls this method.

        Args:
            task_id: The task ID to update
            candidate: The best candidate for this task
        """
        # Get the candidate's score for this specific task
        score = candidate.task_score(task_id) or 0.0
        if score <= 0.0:
            return

        self.task_scores[task_id] = candidate.best_for_task(task_id, self.task_scores.get(task_id, candidate))

    def filter_top(self, accumulator) -> None:
        """Apply accumulator to the best candidate for each task.

        Iterates over all tasks and calls accumulator.append(task_id, candidate) for each best candidate.
        """
        for task_id, best_candidate in self.task_scores.items():
            accumulator.append(task_id, best_candidate)

    def filter_by_task(self, selector) -> List['Candidate']:
        """Delegate filtering to selector with task score data from ScoreMatrix.

        ScoreMatrix provides the best candidates it has stored for each task.
        Selector operates only on the data provided, never accessing pool directly.

        Args:
            selector: Selection strategy that implements filter(task_score_data)

        Returns:
            List of candidates selected by the selector
        """
        # Prepare task score data using ScoreMatrix's stored best candidates
        task_score_data = {}

        for task_id, best_candidate in self.task_scores.items():
            score = best_candidate.task_score(task_id)
            # ScoreMatrix provides best candidate for each task
            task_score_data[task_id] = [(best_candidate, score)]

        # Delegate to selector with prepared data
        return selector.filter(task_score_data)

    def filter_by_candidate(self, selector) -> List['Candidate']:
        """Delegate filtering to selector with array of best-performing candidates.

        ScoreMatrix provides the list of candidates that perform best on at least one task.
        Selector operates only on the candidates array provided.

        Args:
            selector: Selection strategy that implements filter(candidates_array)

        Returns:
            List of candidates selected by the selector
        """
        # Get unique candidates that perform best on at least one task
        unique_candidates = set(self.task_scores.values())
        return selector.filter(list(unique_candidates))
