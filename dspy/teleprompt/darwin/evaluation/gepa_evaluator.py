"""
This module contains specialized evaluators for the GEPA optimization process.
- ParentFastCompare: Quickly validates new candidates against their parents on a minibatch.
- FullTaskScores: Performs a comprehensive evaluation on a full, stable dataset.
These can be chained together to create a multi-phase evaluation pipeline.
"""

import logging
from typing import List, TYPE_CHECKING
import dspy
from .evaluator import Evaluator
from .metrics import Assessor
from ..data.cohort import NewBorns, Survivors
from ..budget import Budget
from .acceptance import StrictImprovementAcceptance
from .cache import EvaluationCache
from .proposal_selection import AllImprovements
from .policy import FullEvaluationPolicy
from .batching import resolve_batch_evaluator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..config import DarwinConfig


class ParentFastCompare(Evaluator):
    """
    An evaluator that performs a quick validation of new candidates (children)
    by comparing them against their parents on a small, minibatch of data.

    This corresponds to Phase 1 of the original GEPA evaluation logic.
    """

    def __init__(self, *, config: "DarwinConfig", minibatch_data: List[dspy.Example] = None,
                 evaluation_cache=None, **kwargs):
        """
        Args:
            minibatch_data: Small validation dataset for quick parent-child comparison.
        """
        super().__init__()
        self.config = config
        self.assessor = config.fitness_function
        self.minibatch_data = minibatch_data or []
        configured_acceptance = config.acceptance_criterion
        acceptance_criterion = configured_acceptance
        self.acceptance_criterion = (
            acceptance_criterion() if isinstance(acceptance_criterion, type)
            else acceptance_criterion or StrictImprovementAcceptance()
        )
        self.evaluation_cache = evaluation_cache or EvaluationCache()
        configured_selection = config.proposal_selection
        proposal_selection = configured_selection
        self.proposal_selection = (
            proposal_selection() if isinstance(proposal_selection, type)
            else proposal_selection or AllImprovements()
        )
        self.verbose = False
        self.last_proposal_records = []

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
        self.last_proposal_records = []
        validation_details = {}
        passed_acceptance = set()

        for candidate in new_borns.candidates:
            is_promising = False
            if not candidate.parents:
                # Initial candidates are always promising.
                is_promising = True
                improvement = float("inf")
                passed_acceptance.add(id(candidate))
            else:
                is_improved, _, improvement = self._validate_on_minibatch(candidate, budget)
                if is_improved:
                    is_promising = True
                    passed_acceptance.add(id(candidate))
                validation_details[id(candidate)] = getattr(
                    self, "_last_validation_details", {}
                )

            if is_promising:
                promising_candidates.append(candidate)
                improvements.append(improvement)

        promising_candidates = self.proposal_selection.select(
            promising_candidates, improvements
        )
        selected_ids = {id(candidate) for candidate in promising_candidates}
        self.last_proposal_records = [
            {
                "candidate_id": id(candidate),
                "parent_ids": [id(parent) for parent in candidate.parents],
                "passed_acceptance": id(candidate) in passed_acceptance,
                "selected": id(candidate) in selected_ids,
                "merge": candidate.creation_metadata.get("merge_type") == "system_aware",
                **validation_details.get(id(candidate), {}),
            }
            for candidate in new_borns.candidates
        ]

        self.publish('parent_fast_compare_summary',
                    {'passed': len(promising_candidates), 'total': len(new_borns.candidates)})
        return Survivors(*promising_candidates, iteration=new_borns.iteration)

    def _validate_on_minibatch(self, child: 'Candidate', budget: Budget) -> tuple[bool, int]:
        """Compares a child and parent on a random minibatch from the development set."""
        minibatch_data = child.proposal_minibatch or self.minibatch_data
        if not minibatch_data:
            return False, 0, float("-inf")

        try:
            cost = len(minibatch_data) * (len(child.parents) + 1)
            comparison_parents = list(child.parents)
            if child.creation_metadata.get("merge_type") == "system_aware":
                # The common ancestor is the merge base, not an additional
                # candidate that the merged program must beat. Official GEPA
                # compares the merged proposal with its two descendants.
                ancestor = child.creation_metadata.get("ancestor_candidate")
                comparison_parents = [
                    parent for parent in comparison_parents if parent is not ancestor
                ]
                if len(comparison_parents) == len(child.parents):
                    comparison_parents = comparison_parents[:2]
                cost = len(minibatch_data) * (len(comparison_parents) + 1)
            if hasattr(budget, "can_spend") and not budget.can_spend("evaluation", cost):
                self._last_validation_details = {"reason": "budget_exhausted", "cost": cost}
                return False, 0, float("-inf")
            # Notify observers about validation start with all relevant info
            self.publish('validate_on_minibatch', child, len(minibatch_data))
            child_scores = self._evaluate(child, minibatch_data, persist_scores=False)
            parent_scores = []
            for parent in comparison_parents:
                scores = self._evaluate(parent, minibatch_data, persist_scores=False)
                parent_scores.append([float(score.value) for score in scores])

            child_values = [float(score.value) for score in child_scores]
            # Crossover proposals must improve every parent they were derived
            # from. This is conservative for multi-parent proposals and is
            # identical to the single-parent GEPA path.
            deltas = [
                sum(child_values) - sum(values)
                for values in parent_scores
            ]
            is_merge = child.creation_metadata.get("merge_type") == "system_aware"
            if is_merge:
                # GEPA crossover is allowed to preserve a parent's score. A
                # merge is accepted when it reaches the better parent, while
                # ordinary mutations retain the configured strict criterion.
                parent_totals = [sum(values) for values in parent_scores]
                child_total = sum(child_values)
                best_parent_total = max(parent_totals, default=float("-inf"))
                is_improved = child_total >= best_parent_total
                improvement = child_total - best_parent_total
            else:
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
            self._last_validation_details = {
                "child_scores": child_values,
                "parent_scores": parent_scores,
                "improvement": improvement,
                "cost": cost,
                "reason": "accepted" if is_improved else "not_improved",
            }

            self.publish('validation_result', child,
                        {'passed': is_improved, 'cost': cost, 'parent_avg': avg_parent, 'child_avg': avg_child})
            return is_improved, cost, improvement

        except Exception as e:
            # Keep this as direct logging since it's an error case
            logger.warning(f"Minibatch validation failed: {e}")
            self._last_validation_details = {
                "reason": "validation_error",
                "error": str(e),
            }
            return False, len(minibatch_data) * (len(comparison_parents) + 1), float("-inf")

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

    def __init__(self, *, config: "DarwinConfig", validation_data: List[dspy.Example] = None,
                 evaluation_cache=None, **kwargs):
        """
        Args:
            validation_data: Full validation dataset for comprehensive evaluation.
        """
        super().__init__()
        self.config = config
        self.assessor = config.fitness_function
        self.validation_data = validation_data or []
        self.evaluation_cache = evaluation_cache or EvaluationCache()
        configured_policy = config.validation_policy
        self.validation_policy = (
            configured_policy() if isinstance(configured_policy, type)
            else configured_policy or FullEvaluationPolicy()
        )
        configured_batch_evaluator = config.batch_evaluator
        self.batch_evaluator = resolve_batch_evaluator(configured_batch_evaluator)
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

        return self._evaluate_with_batch_evaluator(new_borns, budget)

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

    def _evaluate_with_batch_evaluator(self, new_borns: NewBorns, budget: Budget) -> Survivors:
        """Evaluate candidates through the configured batch-evaluator strategy.

        The hook receives ``[(candidate, missing_examples), ...]`` and must
        return metric lists in the same order. Cached examples are omitted.
        The default strategy preserves DSPy's existing per-candidate execution
        semantics; an adapter can replace it with true cross-candidate batching.
        """
        jobs = []
        eval_data_by_candidate = {}
        for candidate in new_borns.candidates:
            eval_data = self.validation_policy.get_eval_batch(
                self.validation_data, iteration=self._iteration, candidate=candidate
            )
            if jobs and hasattr(budget, "can_spend") and not budget.can_spend(
                "evaluation", len(eval_data)
            ):
                break
            eval_data_by_candidate[candidate] = eval_data
            cached = [self.evaluation_cache.get(candidate, example) for example in eval_data]
            missing = [example for example, score in zip(eval_data, cached) if score is None]
            if missing:
                jobs.append((candidate, missing))

        if jobs:
            fresh_by_job = self.batch_evaluator.evaluate(jobs, self.assessor, self)
            if len(fresh_by_job) != len(jobs):
                raise ValueError(
                    "batch_evaluator must return one metric list per evaluation job"
                )
            for (candidate, examples), scores in zip(jobs, fresh_by_job, strict=True):
                if len(scores) != len(examples):
                    raise ValueError(
                        "batch_evaluator returned the wrong number of scores for a job"
                    )
                for example, score in zip(examples, scores, strict=True):
                    self.evaluation_cache.put(candidate, example, score)

        evaluated_candidates = []
        for candidate, eval_data in eval_data_by_candidate.items():
            scores = [self.evaluation_cache.get(candidate, example) for example in eval_data]
            candidate.scores = list(scores)
            self.publish(
                'candidate_evaluation_result', candidate,
                {'average_score': candidate.average_score(), 'scores_count': len(scores)},
            )
            budget.spend_on_evaluation(
                candidate.module,
                {"phase": "full_evaluation", "examples": len(eval_data)},
            )
            evaluated_candidates.append(candidate)

        self.publish('comprehensive_evaluation_complete',
                     {'candidates_count': len(evaluated_candidates), 'tasks_count': len(self.validation_data)})
        self._iteration += 1
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
