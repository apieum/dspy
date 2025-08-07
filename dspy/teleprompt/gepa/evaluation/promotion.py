"""Promotion evaluation implementation."""

from typing import List, Callable
import dspy
from .evaluator import Evaluator
from ..data.cohort import Cohort


class PromotionEvaluator(Evaluator):
    """Evaluator that promotes promising candidates."""

    def __init__(self, metric: Callable,
                 promotion_threshold: float = 0.5):
        self.metric = metric
        self.promotion_threshold = promotion_threshold
        # These will be set during prepare_for_compilation()
        self.evaluation_data: List[dspy.Example] = []
        self.minibatch_data: List[dspy.Example] = []

    def evaluate(self, cohort: Cohort, budget) -> Cohort:
        """Evaluate candidates and filter based on promotion threshold.

        Args:
            cohort: Candidates to evaluate
            budget: Budget to track costs

        Returns:
            Cohort containing only promoted candidates
        """
        promoted_candidates = []

        for candidate in cohort.candidates:
            # Evaluate candidate performance using internal evaluation data
            candidate.evaluate_on_batch(self.evaluation_data, self.metric)

            # Track budget for evaluation
            budget.spend_on_evaluation(candidate.module, {"phase": "evaluation", "examples": len(self.evaluation_data)})

            # Calculate average score from task scores
            avg_score = candidate.average_task_score()

            # Promote if above threshold
            if avg_score >= self.promotion_threshold:
                promoted_candidates.append(candidate)

        filtered_cohort = Cohort(*promoted_candidates)
        return filtered_cohort

    def get_metric(self) -> Callable:
        return self.metric

    def start_compilation(self, student: dspy.Module, 
                         d_feedback: List[dspy.Example], 
                         d_pareto: List[dspy.Example]) -> None:
        """Prepare evaluator with Pareto dataset for candidate evaluation (GEPA Algorithm 1).

        Evaluator uses D_pareto for measuring candidate performance.
        """
        # Evaluator uses D_pareto for final candidate evaluation (not D_feedback)
        self.evaluation_data = d_pareto
        # Create minibatch as 20% of pareto data for fast filtering
        minibatch_size = max(1, len(d_pareto) // 5)
        self.minibatch_data = d_pareto[:minibatch_size]


    def evaluate_for_promotion(self, cohort: Cohort, budget) -> Cohort:
        """Evaluate candidates and return cohort with populated task_scores.

        This method evaluates each candidate on the evaluation data
        and populates their task_scores integration.
        Used for initial candidate setup and direct promotion scenarios.

        Args:
            cohort: Cohort to evaluate
            budget: Budget to track evaluation costs

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

            # Track budget for promotion evaluation
            budget.spend_on_evaluation(candidate.module, {"phase": "promotion", "examples": len(self.evaluation_data)})

        return cohort
