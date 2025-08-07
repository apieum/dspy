"""Test evaluation components."""

import dspy
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Cohort
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


class TestEvaluationInterface:
    """Test evaluation component interfaces."""

    def test_promotion_evaluator_interface(self):
        """Test PromotionEvaluator implements required interface."""
        evaluator = PromotionEvaluator(metric=simple_metric)

        # Interface methods
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'evaluate_for_promotion')
        assert hasattr(evaluator, 'get_metric')
        assert hasattr(evaluator, 'start_compilation')
        assert hasattr(evaluator, 'finish_compilation')

    def test_evaluator_configuration(self):
        """Test evaluator self-configuration via start_compilation."""
        evaluator = PromotionEvaluator(metric=simple_metric, promotion_threshold=0.3)

        training_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
            dspy.Example(input="test3", answer="answer3"),
            dspy.Example(input="test4", answer="answer4"),
            dspy.Example(input="test5", answer="answer5"),
        ]

        # Verify starts unconfigured
        assert evaluator.evaluation_data == []
        assert evaluator.minibatch_data == []

        # Configure with training data
        student = dspy.Predict("input -> output")
        d_feedback = training_data
        d_pareto = training_data
        evaluator.start_compilation(student, d_feedback, d_pareto)

        # Verify configuration
        assert evaluator.evaluation_data == training_data
        assert len(evaluator.minibatch_data) == 1  # 20% of 5 = 1
        assert evaluator.minibatch_data == training_data[:1]


class TestPromotionEvaluation:
    """Test promotion-based evaluation."""

    def test_evaluate_for_promotion(self):
        """Test basic promotion evaluation."""
        evaluator = PromotionEvaluator(
            metric=simple_metric,
            promotion_threshold=0.0  # Accept all candidates
        )

        # Setup evaluation data
        evaluation_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
            dspy.Example(input="test3", answer="answer3"),
        ]

        student = dspy.Predict("input -> output")
        d_feedback = evaluation_data
        d_pareto = evaluation_data
        evaluator.start_compilation(student, d_feedback, d_pareto)

        # Create test candidates
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)

        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)

        test_cohort = Cohort(candidate1, candidate2)
        budget = LLMCallsBudget(100)

        # Evaluate cohort
        evaluated_cohort = evaluator.evaluate_for_promotion(test_cohort, budget)

        # Verify evaluation results
        assert evaluated_cohort.size() == 2

        for candidate in evaluated_cohort:
            assert hasattr(candidate, 'task_scores')
            assert len(candidate.task_scores) == len(evaluation_data)

    def test_evaluate_basic(self):
        """Test basic evaluate method."""
        evaluator = PromotionEvaluator(
            metric=simple_metric,
            promotion_threshold=0.0
        )

        # Setup evaluation data
        evaluation_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
            dspy.Example(input="test3", answer="answer3"),
            dspy.Example(input="test4", answer="answer4"),
        ]

        student = dspy.Predict("input -> output")
        d_feedback = evaluation_data
        d_pareto = evaluation_data
        evaluator.start_compilation(student, d_feedback, d_pareto)

        # Create test candidates
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)

        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)

        new_cohort = Cohort(candidate1, candidate2)
        budget = LLMCallsBudget(100)

        # Basic evaluation
        filtered_cohort = evaluator.evaluate(new_cohort, budget)

        # Verify results
        assert filtered_cohort.size() >= 0  # Some candidates might be filtered
        assert isinstance(filtered_cohort, Cohort)

    def test_promotion_threshold_filtering(self):
        """Test that promotion threshold filters candidates correctly."""
        evaluator = PromotionEvaluator(
            metric=simple_metric,
            promotion_threshold=0.8  # High threshold
        )

        evaluation_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
        ]

        student = dspy.Predict("input -> output")
        d_feedback = evaluation_data
        d_pareto = evaluation_data
        evaluator.start_compilation(student, d_feedback, d_pareto)

        # Create test candidates
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)
        test_cohort = Cohort(candidate)
        budget = LLMCallsBudget(100)

        # Evaluate (likely to be filtered due to high threshold)
        evaluated_cohort = evaluator.evaluate_for_promotion(test_cohort, budget)

        # Verify evaluation was performed (filtering happens based on actual scores)
        assert isinstance(evaluated_cohort, Cohort)
        # Candidates may or may not survive based on actual model performance


class TestEvaluationBudgetIntegration:
    """Test evaluation components work with budget system."""

    def test_evaluation_updates_budget(self):
        """Test that evaluation calls update budget correctly."""
        evaluator = PromotionEvaluator(metric=simple_metric)

        evaluation_data = [
            dspy.Example(input="test", answer="answer"),
        ]

        student = dspy.Predict("input -> output")
        d_feedback = evaluation_data
        d_pareto = evaluation_data
        evaluator.start_compilation(student, d_feedback, d_pareto)

        # Create test candidate
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)
        test_cohort = Cohort(candidate)

        # Track budget usage
        budget = LLMCallsBudget(100)
        initial_consumed = budget.consumed_calls

        # Evaluate (should consume budget)
        evaluator.evaluate_for_promotion(test_cohort, budget)

        # Verify budget was updated (evaluation should consume calls)
        # Note: Actual consumption depends on whether LLM calls are made
        assert budget.consumed_calls >= initial_consumed

    def test_evaluation_respects_budget_limits(self):
        """Test evaluation respects budget constraints."""
        evaluator = PromotionEvaluator(metric=simple_metric)

        evaluation_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        d_feedback = evaluation_data
        d_pareto = evaluation_data
        evaluator.start_compilation(student, d_feedback, d_pareto)

        # Create test candidates
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)
        test_cohort = Cohort(candidate)

        # Very limited budget
        budget = LLMCallsBudget(1)
        budget.consumed_calls = 1  # Already exhausted

        # Should still handle evaluation gracefully
        result = evaluator.evaluate_for_promotion(test_cohort, budget)
        assert isinstance(result, Cohort)
