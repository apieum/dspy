"""Candidate data structure for GEPA optimization."""
from dataclasses import dataclass, field
from typing import Dict, List, Callable
from typing_extensions import Any

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
    creation_metadata: Dict[str, Any] = field(default_factory=dict)
    task_scores: Dict[int, float] = field(default_factory=dict)  # task_scores[task_id] = score

    def __hash__(self) -> int:
        """Make candidates hashable based on object identity."""
        return hash(id(self))

    def __eq__(self, other) -> bool:
        """Compare candidates by object identity."""
        if not isinstance(other, Candidate):
            return False
        return self is other

    def traverse_ancestors(self, callback: Callable[['Candidate', int, Dict[str, str]], bool]) -> None:
        """Traverse ancestors of this candidate."""
        for parent in self.parents:
            if callback(parent, self.generation_number, self.creation_metadata):
                parent.traverse_ancestors(callback)

    def analyse_scores(self, callback: Callable[[int, float], Any], acc=[]) -> Any:
        """Analyse scores of this candidate."""
        for index, score in self.task_scores.items():
            acc.append(callback(index, score))
        return acc

    def set_task_score(self, task_id: int, score: float) -> None:
        """Set score for a specific task during evaluation."""
        self.task_scores[task_id] = score

    def set_task_scores(self, task_scores: Dict[int, float]) -> None:
        """Set scores for multiple tasks."""
        for task_id, score in task_scores.items():
            self.set_task_score(task_id, score)

    def task_score(self, task_id: int) -> float:
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

    def best_overall(self, other: 'Candidate') -> 'Candidate':
        my_avg_score = self.average_task_score()
        other_avg_score = other.average_task_score()
        if my_avg_score > other_avg_score:
            return self
        elif my_avg_score < other_avg_score:
            return other
        elif self.generation_number > other.generation_number:
            return self
        else:
            return other

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
    def best_on_task(self, task_id, candidates:List['Candidate']) -> 'Candidate':
        """Compare a candidate with a list of candidates and return the best one for a specific task.

        Considers both score and generation number (newer wins on ties).

        Args:
            task_id: The task to compare performance on
            candidates: The list of candidates to compare against

        Returns:
            The better candidate for the task (self or other)
        """
        best_candidate = self
        for candidate in candidates:
            best_candidate = best_candidate.best_for_task(task_id, candidate)
        return best_candidate

    def dominate(self, other: 'Candidate') -> bool:
        """Check if this candidate Pareto-dominates another candidate.

        Returns True if this candidate performs at least as well on all tasks
        and strictly better on at least one task (Pareto dominance).

        Args:
            other: The other candidate to compare against

        Returns:
            True if this candidate dominates the other
        """
        at_least_as_good_on_all = True
        strictly_better_on_one = False

        # Use the task_scores keys from either candidate (they should have the same tasks)
        all_task_ids = set(self.task_scores.keys()) | set(other.task_scores.keys())

        for task_id in all_task_ids:
            my_score = self.task_score(task_id) or 0.0
            other_score = other.task_score(task_id) or 0.0

            if my_score < other_score:
                # I'm worse on this task → no domination possible
                at_least_as_good_on_all = False
                break
            elif my_score > other_score:
                # I'm strictly better on this task
                strictly_better_on_one = True
            # else: equal scores → continue checking other tasks

        return at_least_as_good_on_all and strictly_better_on_one

    def _get_all_ancestors(self) -> set['Candidate']:
        """
        Private helper to recursively get all unique ancestors.
        This is an internal implementation detail.
        """
        ancestors = set()
        # Use a stack for iterative depth-first traversal to avoid recursion limits
        to_visit = list(self.parents)
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(parent.parents)
        return ancestors

    def is_descendant_of(self, other: 'Candidate'):
        if other in self.parents:
            return True
        for parent in self.parents:
            if parent.is_descendant_of(other):
                return True
        return False

    def is_ancestor_of(self, other: 'Candidate') -> bool:
        """Check if this candidate is an ancestor of another candidate."""
        return other.is_descendant_of(self)

    def filter_ancestors(self, allowed_ancestors: set['Candidate']) -> set['Candidate']:
        """
        Filters this candidate's ancestors, keeping only those present in a given set.
        """
        my_ancestors = self._get_all_ancestors()
        return my_ancestors.intersection(allowed_ancestors)

    def find_common_ancestors(self, other: 'Candidate') -> set['Candidate']:
        """Find all common ancestors shared with another candidate."""
        my_ancestors = self._get_all_ancestors()
        return other.filter_ancestors(my_ancestors)

    def is_ancestor_of_any(self, candidates: List['Candidate']) -> bool:
        """Check if this candidate is an ancestor of any candidate in the given list."""
        return any(candidate.is_descendant_of(self) for candidate in candidates)
