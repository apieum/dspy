"""CandidatePool for the GEPA optimizer.
Maintains a pool of candidates with integrated task-based scoring.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Callable
import weakref

from .candidate import Candidate
from .cohort import Cohort


class CandidatePool:
    """Framework-aware candidate pool with integrated task-based scoring.

    Manages candidates across generations with integrated task-based scoring
    for fast access to best candidates per task.
    """

    def __init__(self):
        # Primary storage - candidates directly
        self.candidates: List[Candidate] = []
        # Task-based scoring: task_id -> best candidate for that task
        self.task_scores: Dict[int, Candidate] = {}

    def promote(self, cohort: Cohort, budget=None) -> None:
        """Promote a cohort to the pool and update task scores.

        Candidates should already have their task_scores populated from evaluation.
        
        Args:
            cohort: Cohort to promote to the pool
            budget: Optional budget parameter for consistency with core.py calls
        """
        for candidate in cohort.candidates:
            self.candidates.append(candidate)

        # Let cohort update the task scores with best candidates per task
        cohort.update_scores(self)

    def append(self, candidate: Candidate) -> None:
        """Add a candidate to the pool.

        The candidate should already have its task_scores populated from evaluation.
        """
        self.promote(Cohort(candidate))

    def filter_by_task_scores(self, selector, budget=None) -> List[Candidate]:
        """Filter candidates based on their task-by-task score performance.

        Delegates to selector.

        Args:
            selector: Selection strategy that receives task score data
            budget: Optional budget parameter for consistency with core.py calls

        Returns:
            List of candidates selected based on task performance
        """
        return selector.filter(self.task_scores)

    def filter_best_scores(self, selector) -> List[Candidate]:
        """Filter from candidates with the best scores (those who excel in at least one task).

        Args:
            selector: Selection strategy that receives array of best candidates

        Returns:
            List of candidates selected from the high performers
        """
        unique_candidates = set(self.task_scores.values())
        return selector.filter(list(unique_candidates))

    def filter(self, selector) -> List[Candidate]:
        """Filter all candidates in the pool.

        Direct filtering of all candidates without score considerations.

        Args:
            selector: Selection strategy that receives array of all candidates

        Returns:
            List of candidates selected from the entire pool
        """
        return selector.filter(self.candidates)

    def filter_top(self, accumulator) -> None:
        """Apply accumulator to the best candidate per task.

        Iterates over all tasks and adds best candidates to the accumulator cohort.
        """
        for task_id, best_candidate in self.task_scores.items():
            accumulator.add_candidate(best_candidate)

    def update_score(self, task_id: int, candidate: Candidate) -> None:
        """Update the task scores with a single candidate for a specific task.

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

    def size(self) -> int:
        """Total number of candidates in pool."""
        return len(self.candidates)
