"""
This module contains specialized evaluators for the GEPA optimization process.
- ParentFastCompare: Quickly validates new candidates against their parents on a minibatch.
- FullTaskScores: Performs a comprehensive evaluation on a full, stable dataset.
These can be chained together to create a multi-phase evaluation pipeline.
"""

import logging
from typing import List
import dspy
from .evaluator import Evaluator
from .metrics import Assessor
from ..data.cohort import NewBorns, Survivors
from ..budget import Budget

logger = logging.getLogger(__name__)


class ParentFastCompare(Evaluator):
    """
    An evaluator that performs a quick validation of new candidates (children)
    by comparing them against their parents on a small, minibatch of data.

    This corresponds to Phase 1 of the original GEPA evaluation logic.
    """

    def __init__(self, assessor: Assessor, minibatch_data: List[dspy.Example] = None, **kwargs):
        """
        Args:
            assessor: The assessor to evaluate predictions against examples.
            minibatch_data: Small validation dataset for quick parent-child comparison.
        """
        super().__init__()
        self.assessor = assessor
        self.minibatch_data = minibatch_data or []
        self.verbose = False

    def start_compilation(self, student: dspy.Module, dataset_manager=None, verbose: bool = False) -> None:
        """Set verbose mode for the evaluator."""
        self.dataset_manager = dataset_manager
        self.verbose = verbose


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
                is_improved, _ = self._validate_on_minibatch(candidate, budget)
                if is_improved:
                    is_promising = True

            if is_promising:
                promising_candidates.append(candidate)

        self.publish('parent_fast_compare_summary',
                    {'passed': len(promising_candidates), 'total': len(new_borns.candidates)})
        return Survivors(*promising_candidates, iteration=new_borns.iteration)

    def _validate_on_minibatch(self, child: 'Candidate', budget: Budget) -> tuple[bool, int]:
        """Compares a child and parent on a random minibatch from the development set."""
        if not self.minibatch_data:
            return False, 0

        try:
            # Notify observers about validation start with all relevant info
            self.publish('validate_on_minibatch', child, len(self.minibatch_data))
            child_scores = child.evaluate_on_batch(
                self.minibatch_data, assessor=self.assessor, channel=self, persist_scores=False
            )
            parent_avg_scores = []
            for parent in child.parents:
                parent_scores = parent.evaluate_on_batch(
                    self.minibatch_data, assessor=self.assessor, channel=self, persist_scores=False
                )
                parent_avg = sum(float(score.value) for score in parent_scores) / len(parent_scores) if parent_scores else 0.0
                parent_avg_scores.append(parent_avg)

            avg_child = sum(float(score.value) for score in child_scores) / len(child_scores) if child_scores else 0.0
            avg_parent = min(parent_avg_scores) if parent_avg_scores else 0.0 # get the weakest parent score for comparison

            # Allow small tolerance for floating point precision and enable progression when equal
            tolerance = 0.01  # 1% tolerance
            is_improved = (avg_child >= avg_parent - tolerance)  # Accept equal or better scores
            cost = len(self.minibatch_data) * (len(child.parents) + 1)  # Cost for evaluating both child and parents
            budget.spend_on_evaluation(child.module, {"phase": "validation", "cost": cost})

            self.publish('validation_result', child,
                        {'passed': is_improved, 'cost': cost, 'parent_avg': avg_parent, 'child_avg': avg_child})
            return is_improved, cost

        except Exception as e:
            # Keep this as direct logging since it's an error case
            logger.warning(f"Minibatch validation failed: {e}")
            return False, len(self.minibatch_data) * len(child.parents)

class FullTaskScores(Evaluator):
    """
    An evaluator that computes scores for all candidates on the full, stable
    evaluation dataset.

    This corresponds to Phase 2 of the original GEPA evaluation logic. It assumes
    that the candidates it receives have already been validated as promising.
    """

    def __init__(self, assessor: Assessor, validation_data: List[dspy.Example] = None, **kwargs):
        """
        Args:
            assessor: The assessor to evaluate predictions against examples.
            validation_data: Full validation dataset for comprehensive evaluation.
        """
        super().__init__()
        self.assessor = assessor
        self.validation_data = validation_data or []
        self.verbose = False

    def start_compilation(self, student: dspy.Module, dataset_manager=None, verbose: bool=False) -> None:
        """Set verbose mode for the evaluator."""
        self.dataset_manager = dataset_manager
        self.verbose = verbose

    def evaluate(self, new_borns: NewBorns, budget: Budget) -> Survivors:
        """
        Computes and assigns task scores for every candidate in the cohort
        on the full evaluation set.
        """
        if not self.validation_data:
            raise ValueError("Validation data not provided.")

        # Phase 2: Comprehensive evaluation on full validation set
        self.publish('comprehensive_evaluation_start',
                    {'candidates_count': len(new_borns.candidates), 'tasks_count': len(self.validation_data)})

        for candidate in new_borns.candidates:
            # Comprehensive evaluation on complete validation set - now returns List[Metric] directly
            scores = candidate.evaluate_on_batch(self.validation_data, assessor=self.assessor, channel=self)

            # Report final candidate performance
            avg_score = candidate.average_score()
            self.publish('candidate_evaluation_result', candidate, {'average_score': avg_score, 'scores_count': len(scores)})

            # Technical details handled by observers if needed

            budget.spend_on_evaluation(
                candidate.module,
                {"phase": "full_evaluation", "examples": len(self.validation_data)}
            )

        self.publish('comprehensive_evaluation_complete',
                    {'candidates_count': len(new_borns.candidates), 'tasks_count': len(self.validation_data)})
        # All candidates that get a full evaluation are considered "survivors" of this stage.
        return Survivors(*new_borns.to_list(), iteration=new_borns.iteration)

GEPATwoPhasesEval = Evaluator.create_chain("GEPATwoPhasesEval", [ParentFastCompare, FullTaskScores])
