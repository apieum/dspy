"""Candidate data structure for GEPA optimization."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import dspy
from dspy import Module


@dataclass
class Candidate:


    """A candidate solution that encapsulates a DSPy module.
    The candidate knows its lineage (parent relationships)
    """
    module: Module  # DSPy module - the actual program
    parents: List['Candidate'] = field(default_factory=list)  # Direct parent references
    generation_number: int = 0  # Which generation this belongs to
    had_child: bool = False  # Whether this candidate has a child
    creation_metadata: Dict[str, str] = field(default_factory=dict)
    task_scores: Dict[int, float] = field(default_factory=dict)  # task_scores[task_id] = score

    def __hash__(self) -> int:
        """Make candidates hashable based on object identity."""
        return hash(id(self))

    def __eq__(self, other) -> bool:
        """Compare candidates by object identity."""
        if not isinstance(other, Candidate):
            return False
        return self is other

    def set_task_score(self, task_id: int, score: float) -> None:
        """Set score for a specific task during evaluation."""
        self.task_scores[task_id] = score

    def set_task_scores(self, task_scores: Dict[int, float]) -> None:
        """Set scores for multiple tasks."""
        for task_id, score in task_scores.items():
            self.set_task_score(task_id, score)

    def task_score(self, task_id: int) -> Optional[float]:
        """Get score for a specific task."""
        return self.task_scores.get(task_id, 0.0)

    def average_task_score(self) -> float:
        """Calculate average score across all tasks."""
        return sum(self.task_scores.values()) / len(self.task_scores) if self.task_scores else 0.0

    def evaluate_on_example(self, example: dspy.Example, task_id: int, metric: Callable) -> float:
        """Evaluate this candidate on a single example using provided metric."""
        try:
            prediction = self.module(**example.inputs())
            score = float(metric(example, prediction))
            self.task_scores[task_id] = score
            return score
        except Exception:
            return 0.0  # Failed evaluation

    def evaluate_on_batch(self, examples: List[dspy.Example], metric: Callable) -> Dict[int, float]:
        """Evaluate this candidate on a batch of examples."""
        if not examples:
            return {}

        for task_id, example in enumerate(examples):
            self.evaluate_on_example(example, task_id, metric)
        return self.task_scores

    def best_for_task(self, task_id: int, other: 'Candidate') -> 'Candidate':
        """Compare two candidates and return the best one for a specific task.

        Considers both score and generation number (newer wins on ties).

        Args:
            task_id: The task to compare performance on
            other: The other candidate to compare against

        Returns:
            The better candidate for the task (self or other)
        """
        my_score = self.task_score(task_id) or 0.0
        other_score = other.task_score(task_id) or 0.0

        # Better score wins
        if my_score > other_score:
            return self

        # Same score - newer generation wins
        if my_score == other_score and self.generation_number > other.generation_number:
            return self

        return other
