"""Result objects for Darwin optimization."""

from typing import Any, List

import dspy


class Result:
    """Base class for optimization results."""

    pass


class Success(Result):
    """Represents a successful optimization result."""

    def __init__(self, candidates: List[dspy.Module], history: Any):
        self.candidates = candidates
        self.history = history

    def get_best_generalist(self) -> dspy.Module:
        """Returns the candidate with the highest average score."""
        if not self.candidates:
            raise ValueError("No candidates found in the result.")

        best_candidate = max(self.candidates, key=lambda c: c.average_score() if hasattr(c, 'average_score') else -1)
        return best_candidate

    def get_best_specialist(self, task_id: str) -> dspy.Module:
        """Returns the best candidate for a specific task."""
        if not self.candidates:
            raise ValueError("No candidates found in the result.")

        best_candidate = max(
            self.candidates,
            key=lambda c: c.find_score_by_uuid(task_id).value if hasattr(c, 'find_score_by_uuid') and c.find_score_by_uuid(task_id) else -1,
        )
        return best_candidate


class Failure(Result):
    """Represents a failed optimization."""

    def __init__(self, reason: str):
        self.reason = reason


class OptimizationFailureError(Exception):
    """Custom exception for optimization failures."""
    pass
