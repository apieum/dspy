"""Promotion evaluation implementation."""

from typing import List, Callable, Optional, TYPE_CHECKING
import dspy
from .evaluator import Evaluator
from ..data.cohort import Cohort

if TYPE_CHECKING:
    from ..data.candidate_pool import CandidatePool


class PromotionEvaluator(Evaluator):
    """Evaluator that promotes promising candidates."""

    def __init__(self, metric: Callable,
                 promotion_threshold: float = 0.5):
        self.metric = metric
        self.promotion_threshold = promotion_threshold
        # These will be set during prepare_for_compilation()
        self.evaluation_data: List[dspy.Example] = []
        self.minibatch_data: List[dspy.Example] = []

    def evaluate(self, cohort: Cohort) -> Cohort:

        promoted_candidates = []

        for candidate in cohort.candidates:
            # Evaluate candidate performance using internal evaluation data
            candidate.evaluate_on_batch(self.evaluation_data, self.metric)
            
            # Calculate average score from task scores
            avg_score = candidate.average_task_score()

            # Promote if above threshold
            if avg_score >= self.promotion_threshold:
                promoted_candidates.append(candidate)

        filtered_cohort = Cohort(*promoted_candidates)
        return filtered_cohort

    def get_metric(self) -> Callable:
        return self.metric

    def start_compilation(self, student: dspy.Module, training_data: List[dspy.Example]) -> None:
        """Prepare evaluator with training dataset when compilation begins.

        Creates evaluation dataset and minibatch dataset for 2-phase evaluation.
        """
        self.evaluation_data = training_data
        # Create minibatch as 20% of training data for fast filtering
        minibatch_size = max(1, len(training_data) // 5)
        self.minibatch_data = training_data[:minibatch_size]

    def evaluate_two_phase(self, cohort: Cohort) -> Cohort:
        """Implement 2-phase evaluation as described in GEPA paper.

        Fast promotion filter: Quick minibatch evaluation to filter promising candidates
        Full task evaluation: Complete evaluation on main performance data

        Args:
            cohort: New candidates to evaluate

        Returns:
            Cohort with promoted candidates (with task_scores populated)
        """

        # Fast minibatch evaluation to filter promising candidates
        minibatch_promoted = self._fast_promotion_filter(cohort, self.minibatch_data)

        if not minibatch_promoted.candidates:
            return minibatch_promoted

        # Full evaluation on main performance data
        fully_evaluated = self._full_task_evaluation(minibatch_promoted, self.evaluation_data)

        return fully_evaluated

    def _fast_promotion_filter(self, cohort: Cohort, minibatch_data: List[dspy.Example]) -> Cohort:
        """Fast filter using minibatch to identify promising candidates."""
        promoted_candidates = []

        for candidate in cohort.candidates:
            # Quick evaluation on minibatch
            candidate.evaluate_on_batch(minibatch_data, self.metric)
            
            # Calculate average score from task scores
            avg_score = candidate.average_task_score()

            # Promote if above threshold
            if avg_score >= self.promotion_threshold:
                promoted_candidates.append(candidate)

        filtered_cohort = Cohort(*promoted_candidates)
        return filtered_cohort

    def _full_task_evaluation(self, filtered_cohort: Cohort, full_data: List[dspy.Example]) -> Cohort:
        """Full evaluation on main performance data to populate task_scores."""

        for candidate in filtered_cohort.candidates:
            # Evaluate candidate on each task in full_data
            for task_id, example in enumerate(full_data):
                # Get prediction from candidate's module
                try:
                    prediction = candidate.module(**example.inputs())
                    score = self.metric(example, prediction)
                    candidate.task_scores[task_id] = score
                except Exception:
                    # If evaluation fails, assign zero score
                    candidate.task_scores[task_id] = 0.0

        return filtered_cohort

    def evaluate_for_promotion(self, cohort: Cohort) -> Cohort:
        """Evaluate candidates and return cohort with populated task_scores.

        This method evaluates each candidate on the evaluation data
        and populates their task_scores integration.
        Used for initial candidate setup and direct promotion scenarios.

        Returns:
            Cohort with task_scores populated for all candidates
        """
        for candidate in cohort.candidates:
            # Evaluate candidate on each task in evaluation_data
            for task_id, example in enumerate(self.evaluation_data):
                try:
                    prediction = candidate.module(**example.inputs())
                    score = self.metric(example, prediction)
                    candidate.task_scores[task_id] = score
                except Exception:
                    # If evaluation fails, assign zero score
                    candidate.task_scores[task_id] = 0.0

        return cohort
