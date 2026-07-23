"""Comprehensive evaluator for full dataset evaluation.

Generic implementation of comprehensive evaluation on the full validation dataset.
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


class ComprehensiveEvaluator(Evaluator):
    """Comprehensive evaluation on full validation dataset.

    Generic evaluator that:
    - Evaluates candidates on complete validation set
    - Supports evaluation policies (full, minibatch, adaptive)
    - Uses caching to avoid redundant evaluations
    - Supports batch evaluation strategies
    - Tracks iteration count for adaptive policies

    This is a generalized version of GEPA's FullTaskScores that works
    for any optimization algorithm.

    Attributes:
        assessor: Function to evaluate predictions
        validation_data: Full validation dataset
        evaluation_cache: Cache for evaluation results
        validation_policy: Policy for selecting evaluation batch
        batch_evaluator: Strategy for batching evaluations
    """

    def __init__(
        self,
        *,
        config: "DarwinConfig",
        validation_data: Optional[List[dspy.Example]] = None,
        evaluation_cache: Optional[EvaluationCache] = None,
        score_persistence: Optional[Callable[[Candidate, List[Any]], None]] = None,
        **kwargs
    ):
        """Initialize comprehensive evaluator.

        Args:
            config: Darwin configuration
            validation_data: Full validation dataset
            evaluation_cache: Cache for evaluation results
            score_persistence: Optional function to persist scores to candidate
        """
        super().__init__()
        self.config = config
        self.assessor = config.fitness_function
        self.validation_data = validation_data or []

        if evaluation_cache is None:
            raise ValueError("evaluation_cache must be provided")

        self.evaluation_cache = evaluation_cache
        self.validation_policy = config.validation_policy
        self.batch_evaluator = config.batch_evaluator
        self.score_persistence = score_persistence
        self._iteration = 0
        self.verbose = False

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
        """Evaluate candidates on full validation dataset.

        Args:
            new_borns: New candidates to evaluate
            budget: Budget for evaluation

        Returns:
            Survivors containing evaluated candidates
        """
        if not self.validation_data:
            raise ValueError("Validation data not provided")

        self.publish(
            'comprehensive_evaluation_start',
            {
                'candidates_count': len(new_borns.candidates),
                'tasks_count': len(self.validation_data)
            }
        )

        evaluated_candidates = self._evaluate_with_batch_evaluator(
            new_borns,
            budget
        )

        self.publish(
            'comprehensive_evaluation_complete',
            {
                'candidates_count': len(evaluated_candidates),
                'tasks_count': len(self.validation_data)
            }
        )

        self._iteration += 1

        return Survivors(*evaluated_candidates, iteration=new_borns.iteration)

    def _evaluate_with_batch_evaluator(
        self,
        new_borns: NewBorns,
        budget: Budget
    ) -> List[Candidate]:
        """Evaluate candidates using configured batch evaluator.

        Args:
            new_borns: Candidates to evaluate
            budget: Budget for evaluation

        Returns:
            List of evaluated candidates
        """
        jobs = []
        eval_data_by_candidate = {}

        for candidate in new_borns.candidates:
            eval_data = self.validation_policy.get_eval_batch(
                self.validation_data,
                iteration=self._iteration,
                candidate=candidate
            )

            try:
                budget.spend(
                    BudgetEvent(
                        "evaluation",
                        len(eval_data),
                        {
                            "phase": "comprehensive_evaluation",
                            "candidate_id": id(candidate),
                            "allow_overrun": not eval_data_by_candidate
                        }
                    )
                )
            except BudgetExhaustedError:
                break

            eval_data_by_candidate[candidate] = eval_data

            cached = [
                self.evaluation_cache.get(candidate, example)
                for example in eval_data
            ]

            missing = [
                example
                for example, score in zip(eval_data, cached)
                if score is None
            ]

            if missing:
                jobs.append((candidate, missing))

        if jobs:
            fresh_by_job = self.batch_evaluator.evaluate(
                jobs,
                self.assessor,
                self
            )

            if len(fresh_by_job) != len(jobs):
                raise ValueError(
                    "batch_evaluator must return one metric list per evaluation job"
                )

            for (candidate, examples), scores in zip(jobs, fresh_by_job, strict=True):
                if len(scores) != len(examples):
                    raise ValueError(
                        "batch_evaluator returned wrong number of scores for a job"
                    )

                for example, score in zip(examples, scores, strict=True):
                    self.evaluation_cache.put(candidate, example, score)

        evaluated_candidates = []

        for candidate, eval_data in eval_data_by_candidate.items():
            scores = [
                self.evaluation_cache.get(candidate, example)
                for example in eval_data
            ]

            if self.score_persistence:
                self.score_persistence(candidate, scores)

            avg_score = (
                sum(float(s.value) for s in scores) / len(scores)
                if scores else 0.0
            )

            self.publish(
                'candidate_evaluation_result',
                candidate,
                {
                    'average_score': avg_score,
                    'scores_count': len(scores)
                }
            )

            evaluated_candidates.append(candidate)

        return evaluated_candidates
