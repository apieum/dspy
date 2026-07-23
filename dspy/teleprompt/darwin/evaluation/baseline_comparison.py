"""Baseline comparison evaluator for quick candidate validation.

Generic implementation of quick validation by comparing candidates against
a baseline (e.g., parent, previous best) on a small minibatch.
"""

import logging
from typing import List, TYPE_CHECKING, Optional, Callable, Any
import dspy

from .evaluator import Evaluator
from .metrics import Assessor
from .cache import EvaluationCache
from ..data.cohort import NewBorns, Survivors
from ..data.candidate import Candidate
from ..budget import Budget, BudgetEvent, BudgetExhaustedError

if TYPE_CHECKING:
    from ..config import DarwinConfig


logger = logging.getLogger(__name__)


class BaselineComparison(Evaluator):
    """Quick validation by comparing candidates against baseline on minibatch.

    Generic evaluator that:
    - Compares new candidates against a baseline (parent, previous best, etc.)
    - Uses small minibatch for fast evaluation
    - Filters candidates based on acceptance criterion
    - Supports caching to avoid redundant evaluations

    This is a generalized version of GEPA's ParentFastCompare that works
    for any optimization algorithm.

    Attributes:
        assessor: Function to evaluate predictions
        minibatch_data: Small validation dataset for quick comparison
        acceptance_criterion: Criterion to determine if candidate is better
        evaluation_cache: Cache for evaluation results
        baseline_selector: Function to select baseline for each candidate
    """

    def __init__(
        self,
        *,
        config: "DarwinConfig",
        minibatch_data: Optional[List[dspy.Example]] = None,
        evaluation_cache: Optional[EvaluationCache] = None,
        baseline_selector: Optional[Callable[[Candidate], List[Candidate]]] = None,
        **kwargs
    ):
        """Initialize baseline comparison evaluator.

        Args:
            config: Darwin configuration
            minibatch_data: Small validation dataset
            evaluation_cache: Cache for evaluation results
            baseline_selector: Function to select baseline(s) for each candidate
                             Default: uses candidate.parents
        """
        super().__init__()
        self.config = config
        self.assessor = config.fitness_function
        self.minibatch_data = minibatch_data or []
        self.acceptance_criterion = config.acceptance_criterion

        if evaluation_cache is None:
            raise ValueError("evaluation_cache must be provided")

        self.evaluation_cache = evaluation_cache
        self.baseline_selector = baseline_selector or self._default_baseline_selector
        self.proposal_selection = config.proposal_selection
        self.verbose = False

    @staticmethod
    def _default_baseline_selector(candidate: Candidate) -> List[Candidate]:
        """Default baseline selector returns candidate's parents."""
        return list(candidate.parents) if candidate.parents else []

    def start_compilation(
        self,
        student: dspy.Module,
        dataset_manager=None,
        verbose: bool = False
    ) -> None:
        """Set verbose mode and dataset manager.

        Args:
            student: Program being optimized
            dataset_manager: Dataset manager
            verbose: Enable verbose logging
        """
        self.dataset_manager = dataset_manager
        self.verbose = verbose

    def evaluate(self, new_borns: NewBorns, budget: Budget) -> Survivors:
        """Filter candidates by comparing against baseline on minibatch.

        Args:
            new_borns: New candidates to evaluate
            budget: Budget for evaluation

        Returns:
            Survivors containing promising candidates
        """
        promising_candidates = []
        improvements = []

        for candidate in new_borns.candidates:
            baselines = self.baseline_selector(candidate)

            if not baselines:
                promising_candidates.append(candidate)
                improvements.append(float("inf"))
                continue

            is_improved, improvement = self._compare_to_baselines(
                candidate,
                baselines,
                budget
            )

            if is_improved:
                promising_candidates.append(candidate)
                improvements.append(improvement)

        promising_candidates = self.proposal_selection.select(
            promising_candidates,
            improvements
        )

        self.publish(
            'baseline_comparison_summary',
            {
                'passed': len(promising_candidates),
                'total': len(new_borns.candidates)
            }
        )

        return Survivors(*promising_candidates, iteration=new_borns.iteration)

    def _compare_to_baselines(
        self,
        candidate: Candidate,
        baselines: List[Candidate],
        budget: Budget
    ) -> tuple[bool, float]:
        """Compare candidate against baseline candidates on minibatch.

        Args:
            candidate: Candidate to evaluate
            baselines: Baseline candidates to compare against
            budget: Budget for evaluation

        Returns:
            Tuple of (is_improved, improvement_score)
        """
        if not self.minibatch_data:
            return False, float("-inf")

        try:
            cost = len(self.minibatch_data) * (len(baselines) + 1)

            try:
                budget.spend(
                    BudgetEvent(
                        "evaluation",
                        cost,
                        {
                            "phase": "baseline_validation",
                            "candidate_id": id(candidate)
                        }
                    )
                )
            except BudgetExhaustedError:
                return False, float("-inf")

            self.publish('validate_on_minibatch', candidate, len(self.minibatch_data))

            candidate_scores = self._evaluate_candidate(
                candidate,
                self.minibatch_data
            )

            baseline_scores_list = []
            for baseline in baselines:
                scores = self._evaluate_candidate(baseline, self.minibatch_data)
                baseline_scores_list.append([float(score.value) for score in scores])

            candidate_values = [float(score.value) for score in candidate_scores]

            deltas = [
                sum(candidate_values) - sum(baseline_values)
                for baseline_values in baseline_scores_list
            ]

            is_improved = all(
                self.acceptance_criterion.should_accept(candidate_values, baseline_values)
                for baseline_values in baseline_scores_list
            )

            improvement = min(deltas, default=float("-inf"))

            avg_candidate = sum(candidate_values) / len(candidate_values) if candidate_values else 0.0
            avg_baseline = min(
                (sum(values) / len(values) for values in baseline_scores_list if values),
                default=0.0
            )

            self.publish(
                'validation_result',
                candidate,
                {
                    'passed': is_improved,
                    'cost': cost,
                    'baseline_avg': avg_baseline,
                    'candidate_avg': avg_candidate
                }
            )

            return is_improved, improvement

        except Exception as e:
            logger.warning(f"Minibatch validation failed: {e}")
            return False, float("-inf")

    def _evaluate_candidate(
        self,
        candidate: Candidate,
        examples: List[dspy.Example]
    ) -> List[Any]:
        """Evaluate candidate on examples with caching.

        Args:
            candidate: Candidate to evaluate
            examples: Examples to evaluate on

        Returns:
            List of metric scores
        """
        cached = [
            self.evaluation_cache.get(candidate, example)
            for example in examples
        ]

        missing = [
            example
            for example, score in zip(examples, cached)
            if score is None
        ]

        if missing:
            fresh = self._evaluate_batch(candidate, missing)

            for example, score in zip(missing, fresh):
                self.evaluation_cache.put(candidate, example, score)

        return [
            self.evaluation_cache.get(candidate, example)
            for example in examples
        ]

    def _evaluate_batch(
        self,
        candidate: Candidate,
        examples: List[dspy.Example]
    ) -> List[Any]:
        """Evaluate candidate on batch of examples.

        This method should be overridden for algorithm-specific evaluation.
        Default implementation assumes candidate has evaluate_on_batch method.

        Args:
            candidate: Candidate to evaluate
            examples: Examples to evaluate on

        Returns:
            List of metric scores
        """
        if hasattr(candidate, 'evaluate_on_batch'):
            return candidate.evaluate_on_batch(
                examples,
                assessor=self.assessor,
                channel=self,
                persist_scores=False
            )
        else:
            raise NotImplementedError(
                "Candidate must have evaluate_on_batch method or "
                "_evaluate_batch must be overridden"
            )
