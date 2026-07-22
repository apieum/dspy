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
from .acceptance import StrictImprovementAcceptance
from .cache import EvaluationCache
from .proposal_selection import AllImprovements
from .policy import FullEvaluationPolicy

logger = logging.getLogger(__name__)


class ParentFastCompare(Evaluator):
    """
    An evaluator that performs a quick validation of new candidates (children)
    by comparing them against their parents on a small, minibatch of data.

    This corresponds to Phase 1 of the original GEPA evaluation logic.
    """

    def __init__(self, assessor: Assessor, minibatch_data: List[dspy.Example] = None,
                 acceptance_criterion=None, proposal_selection=None,
                 evaluation_cache=None, **kwargs):
        """
        Args:
            assessor: The assessor to evaluate predictions against examples.
            minibatch_data: Small validation dataset for quick parent-child comparison.
        """
        super().__init__()
        self.assessor = assessor
        self.minibatch_data = minibatch_data or []
        self.acceptance_criterion = acceptance_criterion or StrictImprovementAcceptance()
        self.evaluation_cache = evaluation_cache or EvaluationCache()
        self.proposal_selection = proposal_selection or AllImprovements()
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
        improvements = []

        for candidate in new_borns.candidates:
            is_promising = False
            if not candidate.parents:
                # Initial candidates are always promising.
                is_promising = True
                improvement = float("inf")
            else:
                is_improved, _, improvement = self._validate_on_minibatch(candidate, budget)
                if is_improved:
                    is_promising = True

            if is_promising:
                promising_candidates.append(candidate)
                improvements.append(improvement)

        promising_candidates = self.proposal_selection.select(
            promising_candidates, improvements
        )

        self.publish('parent_fast_compare_summary',
                    {'passed': len(promising_candidates), 'total': len(new_borns.candidates)})
        return Survivors(*promising_candidates, iteration=new_borns.iteration)

    def _validate_on_minibatch(self, child: 'Candidate', budget: Budget) -> tuple[bool, int]:
        """Compares a child and parent on a random minibatch from the development set."""
        if not self.minibatch_data:
            return False, 0, float("-inf")

        try:
            cost = len(self.minibatch_data) * (len(child.parents) + 1)
            if hasattr(budget, "can_spend") and not budget.can_spend("evaluation", cost):
                return False, 0, float("-inf")
            # Notify observers about validation start with all relevant info
            self.publish('validate_on_minibatch', child, len(self.minibatch_data))
            child_scores = self._evaluate(child, self.minibatch_data, persist_scores=False)
            parent_scores = []
            for parent in child.parents:
                scores = self._evaluate(parent, self.minibatch_data, persist_scores=False)
                parent_scores.append([float(score.value) for score in scores])

            child_values = [float(score.value) for score in child_scores]
            # Crossover proposals must improve every parent they were derived
            # from. This is conservative for multi-parent proposals and is
            # identical to the single-parent GEPA path.
            deltas = [
                sum(child_values) - sum(values)
                for values in parent_scores
            ]
            is_improved = all(
                self.acceptance_criterion.should_accept(child_values, values)
                for values in parent_scores
            )
            improvement = min(deltas, default=float("-inf"))
            avg_child = sum(child_values) / len(child_values) if child_values else 0.0
            avg_parent = min(
                (sum(values) / len(values) for values in parent_scores if values),
                default=0.0,
            )
            budget.spend_on_evaluation(child.module, {"phase": "validation", "cost": cost})

            self.publish('validation_result', child,
                        {'passed': is_improved, 'cost': cost, 'parent_avg': avg_parent, 'child_avg': avg_child})
            return is_improved, cost, improvement

        except Exception as e:
            # Keep this as direct logging since it's an error case
            logger.warning(f"Minibatch validation failed: {e}")
            return False, len(self.minibatch_data) * len(child.parents), float("-inf")

    def _evaluate(self, candidate, examples, *, persist_scores):
        cached = [self.evaluation_cache.get(candidate, example) for example in examples]
        missing = [example for example, score in zip(examples, cached) if score is None]
        if missing:
            fresh = candidate.evaluate_on_batch(
                missing, assessor=self.assessor, channel=self,
                persist_scores=persist_scores,
            )
            for example, score in zip(missing, fresh):
                self.evaluation_cache.put(candidate, example, score)
        return [self.evaluation_cache.get(candidate, example) for example in examples]

class FullTaskScores(Evaluator):
    """
    An evaluator that computes scores for all candidates on the full, stable
    evaluation dataset.

    This corresponds to Phase 2 of the original GEPA evaluation logic. It assumes
    that the candidates it receives have already been validated as promising.
    """

    def __init__(self, assessor: Assessor, validation_data: List[dspy.Example] = None,
                 evaluation_cache=None, validation_policy=None, **kwargs):
        """
        Args:
            assessor: The assessor to evaluate predictions against examples.
            validation_data: Full validation dataset for comprehensive evaluation.
        """
        super().__init__()
        self.assessor = assessor
        self.validation_data = validation_data or []
        self.evaluation_cache = evaluation_cache or EvaluationCache()
        self.validation_policy = validation_policy or FullEvaluationPolicy()
        self._iteration = 0
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

        evaluated_candidates = []
        for candidate in new_borns.candidates:
            eval_data = self.validation_policy.get_eval_batch(
                self.validation_data, iteration=self._iteration, candidate=candidate
            )
            # Always score the seed program so a constrained run still has a
            # valid result. Subsequent proposals must fit the evaluation
            # domain budget before they are evaluated.
            if evaluated_candidates or candidate.parents:
                budget_exhausted = hasattr(budget, "can_spend") and not budget.can_spend(
                "evaluation", len(eval_data)
                )
                if budget_exhausted:
                    break
            # Comprehensive evaluation on complete validation set - now returns List[Metric] directly
            scores = self._evaluate(candidate, eval_data)

            # Report final candidate performance
            avg_score = candidate.average_score()
            self.publish('candidate_evaluation_result', candidate, {'average_score': avg_score, 'scores_count': len(scores)})

            # Technical details handled by observers if needed

            budget.spend_on_evaluation(
                candidate.module,
                {"phase": "full_evaluation", "examples": len(eval_data)}
            )
            evaluated_candidates.append(candidate)

        self.publish('comprehensive_evaluation_complete',
                    {'candidates_count': len(new_borns.candidates), 'tasks_count': len(self.validation_data)})
        self._iteration += 1
        # All candidates that get a full evaluation are considered "survivors" of this stage.
        return Survivors(*evaluated_candidates, iteration=new_borns.iteration)

    def _evaluate(self, candidate, examples):
        cached = [self.evaluation_cache.get(candidate, example) for example in examples]
        missing = [example for example, score in zip(examples, cached) if score is None]
        if missing:
            fresh = candidate.evaluate_on_batch(
                missing, assessor=self.assessor, channel=self, persist_scores=True
            )
            for example, score in zip(missing, fresh):
                self.evaluation_cache.put(candidate, example, score)
        scores = [self.evaluation_cache.get(candidate, example) for example in examples]
        # Keep Candidate.scores synchronized for Pareto selection when every
        # score is already cached.
        candidate.scores = list(scores)
        return scores

GEPATwoPhasesEval = Evaluator.create_chain("GEPATwoPhasesEval", [ParentFastCompare, FullTaskScores])
