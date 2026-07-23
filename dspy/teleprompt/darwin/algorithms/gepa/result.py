"""GEPA result objects."""

from typing import Any, Dict, List, Optional, Set

import dspy
from ...result import Result, OptimizationFailureError


class Success(Result):
    """Represents a successful optimization result."""

    def __init__(
        self,
        candidates: List[dspy.Module],
        history: Any,
        *,
        best_candidate=None,
        parents: Optional[Dict[Any, List[Any]]] = None,
        val_subscores: Optional[Dict[Any, Dict[str, float]]] = None,
        per_val_instance_best_candidates: Optional[Dict[str, Set[Any]]] = None,
    ):
        self.candidates = candidates
        self.history = history
        self.best_candidate = best_candidate
        self.parents = parents or {}
        self.val_subscores = val_subscores or {}
        self.per_val_instance_best_candidates = per_val_instance_best_candidates or {}

    def get_best_generalist(self) -> dspy.Module:
        """Returns the candidate with the highest average score."""
        if not self.candidates:
            raise ValueError("No candidates found in the result.")

        return self.best_candidate or max(
            self.candidates,
            key=lambda c: c.average_score() if hasattr(c, 'average_score') else -1,
        )

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
