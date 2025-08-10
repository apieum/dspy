"""
This module contains specialized evaluators for the GEPA optimization process.
- ParentFastCompare: Quickly validates new candidates against their parents on a minibatch.
- FullTaskScores: Performs a comprehensive evaluation on a full, stable dataset.
These can be chained together to create a multi-phase evaluation pipeline.
"""

import logging
from typing import Callable, Optional, TYPE_CHECKING
from dspy import Module
from .evaluator import Evaluator
from ..data.cohort import NewBorns, Survivors
from ..budget import Budget

if TYPE_CHECKING:
    from ..dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class ParentFastCompare(Evaluator):
    """
    An evaluator that performs a quick validation of new candidates (children)
    by comparing them against their parents on a small, random minibatch of data.

    This corresponds to Phase 1 of the original GEPA evaluation logic.
    """

    def __init__(self, metric: Callable, minibatch_size: int = 3, **kwargs):
        """
        Args:
            metric: The function to evaluate predictions against examples.
            minibatch_size: The number of examples to use for the validation minibatch.
        """
        super().__init__()
        self.metric = metric
        self.minibatch_size = minibatch_size
        self.dataset_manager: Optional["DatasetManager"] = None

    def start_compilation(self, student: Module, dataset_manager: "DatasetManager") -> None:
        """Prepares the evaluator with the dataset manager."""
        self.dataset_manager = dataset_manager

    def evaluate(self, new_borns: NewBorns, budget: Budget) -> Survivors:
        """
        Filters a cohort of new candidates, keeping only those that are "promising."

        A candidate is promising if:
        1. It has no parents (i.e., it's an initial candidate).
        2. It outperforms its parent on a random validation minibatch.
        """
        promising_candidates = []

        for candidate in new_borns.candidates:
            is_promising = False
            if not candidate.parents:
                # Initial candidates are always promising.
                is_promising = True
            else:
                is_improved, cost = self._validate_on_minibatch(candidate)
                budget.spend_on_evaluation(candidate.module, {"phase": "validation", "cost": cost})
                if is_improved:
                    is_promising = True

            if is_promising:
                promising_candidates.append(candidate)

        logger.info(f"ParentFastCompare: {len(promising_candidates)}/{len(new_borns.candidates)} candidates passed validation.")
        return Survivors(*promising_candidates, iteration=new_borns.iteration)

    def _validate_on_minibatch(self, child: 'Candidate') -> tuple[bool, int]:
        """Compares a child and parent on a random minibatch from the development set."""
        validation_minibatch = self.dataset_manager.get_validation_minibatch(self.minibatch_size)
        if not validation_minibatch:
            return False, 0

        try:
            # Use the efficient, parallel batch evaluation with dictionary (preserves task_ids)
            child_scores = child.evaluate_on_batch(validation_minibatch, metric=self.metric)
            parent_scores = []
            for parent in child.parents:
                score = parent.evaluate_on_batch(validation_minibatch, metric=self.metric)
                parent_scores.append(sum(score.values()) / len(score) if score else 0.0)

            avg_child = sum(child_scores.values()) / len(child_scores) if child_scores else 0
            avg_parent = min(parent_scores) if parent_scores else 0.0

            is_improved = avg_child > avg_parent
            cost = len(validation_minibatch) * len(child.parents)  # Cost for evaluating both child and parents

            logger.debug(f"Minibatch validation: parent={avg_parent:.3f}, child={avg_child:.3f}, improved={is_improved}")
            return is_improved, cost

        except Exception as e:
            logger.warning(f"Minibatch validation failed: {e}")
            return False, len(validation_minibatch) * len(child.parents)


class FullTaskScores(Evaluator):
    """
    An evaluator that computes scores for all candidates on the full, stable
    evaluation dataset.

    This corresponds to Phase 2 of the original GEPA evaluation logic. It assumes
    that the candidates it receives have already been validated as promising.
    """

    def __init__(self, metric: Callable, **kwargs):
        """
        Args:
            metric: The function to evaluate predictions against examples.
        """
        super().__init__()
        self.metric = metric
        self.dataset_manager: Optional["DatasetManager"] = None

    def start_compilation(self, student: Module, dataset_manager: "DatasetManager") -> None:
        """Prepares the evaluator with the dataset manager."""
        self.dataset_manager = dataset_manager

    def evaluate(self, new_borns: NewBorns, budget: Budget) -> Survivors:
        """
        Computes and assigns task scores for every candidate in the cohort
        on the full evaluation set.
        """
        if not self.dataset_manager:
            raise ValueError("Dataset manager not initialized.")

        evaluation_set = self.dataset_manager.get_eval_set()

        for candidate in new_borns.candidates:
            # Use the efficient, parallel batch evaluation
            candidate.batch_task_scores(evaluation_set, metric=self.metric)
            budget.spend_on_evaluation(
                candidate.module,
                {"phase": "full_evaluation", "examples": len(evaluation_set)}
            )

        logger.info(f"FullTaskScores: Completed full evaluation for {len(new_borns.candidates)} candidates.")
        # All candidates that get a full evaluation are considered "survivors" of this stage.
        return Survivors(*new_borns.to_list(), iteration=new_borns.iteration)

GEPATwoPhasesEval = Evaluator.create_chain("GEPATwoPhasesEval", [ParentFastCompare, FullTaskScores])
