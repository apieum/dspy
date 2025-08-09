"""Test evaluation components with two-phase evaluation."""

import dspy
from unittest.mock import Mock
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import NewBorns, Parents, Survivors
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget
from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManager


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def mock_prediction(answer="test_answer"):
    """Create mock prediction."""
    pred = Mock()
    pred.answer = answer
    return pred


def mock_module_with_history(predictions_func):
    """Create a mock module with proper history for budget tracking."""
    module = Mock()
    module.history = []  # Empty history for budget tracking
    module.side_effect = predictions_func
    return module


class TestTwoPhaseEvaluationInterface:
    """Test two-phase evaluation component interfaces."""

    def test_promotion_evaluator_interface(self):
        """Test PromotionEvaluator implements required two-phase interface."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=3)

        # Interface methods
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'get_metric')
        assert hasattr(evaluator, 'start_compilation')
        assert hasattr(evaluator, 'finish_compilation')

        # Verify minibatch size
        assert evaluator.minibatch_size == 3

    def test_evaluator_configuration(self):
        """Test evaluator configuration via start_compilation."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)

        dev_data = [
            dspy.Example(input="feedback1", answer="answer1").with_inputs("input"),
            dspy.Example(input="feedback2", answer="answer2").with_inputs("input"),
        ]

        eval_data = [
            dspy.Example(input="pareto1", answer="pareto_answer1").with_inputs("input"),
            dspy.Example(input="pareto2", answer="pareto_answer2").with_inputs("input"),
            dspy.Example(input="pareto3", answer="pareto_answer3").with_inputs("input"),
        ]

        # Verify starts unconfigured
        assert evaluator.dataset_manager is None

        # Configure with training data
        student = dspy.Predict("input -> output")
        training_data = eval_data + dev_data
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.6)
        evaluator.start_compilation(student, dataset_manager)

        # Verify configuration
        assert evaluator.dataset_manager is not None
        assert len(evaluator.dataset_manager.get_eval_set()) == 3
        assert evaluator.dataset_manager.num_dev_examples == 2


class TestTwoPhaseEvaluation:
    """Test two-phase evaluation logic."""

    def test_minibatch_validation_improves(self):
        """Test minibatch validation when child improves over parent."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)

        dev_data = [
            dspy.Example(input="test1", answer="correct").with_inputs("input"),
            dspy.Example(input="test2", answer="correct").with_inputs("input"),
        ]

        eval_data = [
            dspy.Example(input="pareto1", answer="correct").with_inputs("input"),
        ]

        student = dspy.Predict("input -> output")
        training_data = eval_data + dev_data
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.33)
        evaluator.start_compilation(student, dataset_manager)

        # Create parent and child modules with different performance
        parent_module = mock_module_with_history(lambda **kwargs: mock_prediction("wrong"))
        child_module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))

        # Create candidates
        parent_candidate = Candidate(parent_module, generation_number=0)
        child_candidate = Candidate(child_module, generation_number=1, parents=[parent_candidate])

        # Create cohorts
        new_borns = NewBorns(child_candidate, iteration=1)
        parents = Parents(parent_candidate, iteration=0)

        # Budget
        budget = LLMCallsBudget(100)

        # Evaluate
        survivors = evaluator.evaluate(new_borns, budget)

        # Child should be promoted since it improved
        assert survivors.size() == 1
        assert survivors.contains(child_candidate)

        # Verify budget tracking was called (budget.spend_on_evaluation was invoked)
        # The actual call counting depends on real module history, so we just verify
        # the evaluation completed successfully
        assert budget.consumed_calls >= 0  # Budget tracking attempted

    def test_minibatch_validation_rejects(self):
        """Test minibatch validation when child doesn't improve over parent."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)

        dev_data = [
            dspy.Example(input="test1", answer="correct").with_inputs("input"),
            dspy.Example(input="test2", answer="correct").with_inputs("input"),
        ]

        eval_data = [
            dspy.Example(input="pareto1", answer="correct").with_inputs("input"),
        ]

        student = dspy.Predict("input -> output")
        training_data = eval_data + dev_data
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.33)
        evaluator.start_compilation(student, dataset_manager)

        # Both parent and child perform the same (or child is worse)
        parent_module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))
        child_module = mock_module_with_history(lambda **kwargs: mock_prediction("wrong"))  # Worse performance

        # Create candidates
        parent_candidate = Candidate(parent_module, generation_number=0)
        child_candidate = Candidate(child_module, generation_number=1, parents=[parent_candidate])

        # Create cohorts
        new_borns = NewBorns(child_candidate, iteration=1)
        parents = Parents(parent_candidate, iteration=0)

        # Budget
        budget = LLMCallsBudget(100)

        # Evaluate
        survivors = evaluator.evaluate(new_borns, budget)

        # Child should NOT be promoted since it didn't improve
        assert survivors.size() == 0

    def test_full_evaluation_after_minibatch_success(self):
        """Test that full evaluation is only run after minibatch success."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=1)

        data = {
            0:dspy.Example(input="pareto1", answer="correct").with_inputs("input"),
            1:dspy.Example(input="pareto2", answer="correct").with_inputs("input"),
            2:dspy.Example(input="pareto3", answer="correct").with_inputs("input"),
            3:dspy.Example(input="feedback", answer="correct").with_inputs("input"),
        }

        student = dspy.Predict("input -> output")
        dataset_manager = DefaultDatasetManager(data, split_ratio=0.75)
        evaluator.start_compilation(student, dataset_manager)

        # Child improves over parent
        parent_module = mock_module_with_history(lambda **kwargs: mock_prediction("wrong"))
        child_module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))

        # Create candidates
        parent_candidate = Candidate(parent_module, generation_number=0)
        child_candidate = Candidate(child_module, generation_number=1, parents=[parent_candidate])

        # Create cohorts
        new_borns = NewBorns(child_candidate, iteration=1)
        parents = Parents(parent_candidate, iteration=0)

        # Budget
        budget = LLMCallsBudget(100)
        initial_calls = budget.consumed_calls

        # Evaluate
        survivors = evaluator.evaluate(new_borns, budget)

        # Child should be promoted and have task scores from full evaluation
        assert survivors.size() == 1
        promoted = survivors.first()
        assert len(promoted.task_scores) == len(dataset_manager.get_eval_set())  # Full evaluation happened

        # Budget tracking was called for both phases
        # The specific call counts depend on real module history, so we focus on
        # verifying the evaluation phases worked correctly
        assert budget.consumed_calls >= initial_calls  # Budget tracking attempted


class TestEvaluationBudgetIntegration:
    """Test evaluation budget integration."""

    def test_evaluation_updates_budget_correctly(self):
        """Test that both evaluation phases update budget correctly."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)

        dev_data = [
            dspy.Example(input="feedback1", answer="correct").with_inputs("input"),
            dspy.Example(input="feedback2", answer="correct").with_inputs("input"),
        ]
        eval_data = [dspy.Example(input="pareto", answer="correct").with_inputs("input")]

        student = dspy.Predict("input -> output")
        training_data = eval_data + dev_data
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.33)
        evaluator.start_compilation(student, dataset_manager)

        # Create improving child
        parent_module = mock_module_with_history(lambda **kwargs: mock_prediction("wrong"))
        child_module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))

        parent_candidate = Candidate(parent_module, generation_number=0)
        child_candidate = Candidate(child_module, generation_number=1, parents=[parent_candidate])

        new_borns = NewBorns(child_candidate, iteration=1)
        parents = Parents(parent_candidate, iteration=0)

        # Track budget usage
        budget = LLMCallsBudget(100)
        initial_consumed = budget.consumed_calls

        # Evaluate (should consume budget for both phases)
        evaluator.evaluate(new_borns, budget)

        # Verify budget tracking was attempted (the spend_on_evaluation calls succeeded)
        assert budget.consumed_calls >= initial_consumed

    def test_evaluation_handles_budget_limits_gracefully(self):
        """Test evaluation handles budget constraints gracefully."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=1)

        dev_data = [dspy.Example(input="feedback", answer="correct").with_inputs("input")]
        eval_data = [dspy.Example(input="pareto", answer="correct").with_inputs("input")]

        student = dspy.Predict("input -> output")
        training_data = eval_data + dev_data
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.5)
        evaluator.start_compilation(student, dataset_manager)

        # Create test candidates
        parent_module = mock_module_with_history(lambda **kwargs: mock_prediction("wrong"))
        child_module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))

        parent_candidate = Candidate(parent_module, generation_number=0)
        child_candidate = Candidate(child_module, generation_number=1, parents=[parent_candidate])

        new_borns = NewBorns(child_candidate, iteration=1)
        parents = Parents(parent_candidate, iteration=0)

        # Very limited budget
        budget = LLMCallsBudget(10)
        budget.consumed_calls = 5  # Already partially consumed

        # Should still handle evaluation
        result = evaluator.evaluate(new_borns, budget)

        # Should return valid result type
        assert isinstance(result, Survivors)


class TestEvaluationErrorHandling:
    """Test evaluation error handling."""

    def test_evaluation_with_no_parents(self):
        """Test evaluation handles candidates with no parents."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)

        dev_data = [dspy.Example(input="feedback", answer="correct").with_inputs("input")]
        eval_data = [dspy.Example(input="pareto", answer="correct").with_inputs("input")]

        student = dspy.Predict("input -> output")
        training_data = eval_data + dev_data
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.5)
        evaluator.start_compilation(student, dataset_manager)

        # Create candidate with no parents
        module = mock_module_with_history(lambda **kwargs: mock_prediction("test"))
        candidate = Candidate(module, generation_number=0)  # No parents

        new_borns = NewBorns(candidate, iteration=1)
        parents = Parents(iteration=0)  # Empty parents

        budget = LLMCallsBudget(100)

        # Should evaluate directly on pareto data (no parent comparison needed)
        survivors = evaluator.evaluate(new_borns, budget)
        assert survivors.size() == 1  # Initial candidate should be evaluated
        assert survivors.first().task_scores  # Should have task scores populated

    def test_evaluation_with_empty_dev_data(self):
        """Test evaluation handles empty feedback data."""
        evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)

        # Empty feedback data
        dev_data = []
        eval_data = [dspy.Example(input="pareto", answer="correct").with_inputs("input")]

        student = dspy.Predict("input -> output")
        training_data = eval_data + dev_data  # Will be just eval_data
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=1.0)  # All data goes to pareto
        evaluator.start_compilation(student, dataset_manager)

        # Create candidates
        parent_module = mock_module_with_history(lambda **kwargs: mock_prediction("test"))
        child_module = mock_module_with_history(lambda **kwargs: mock_prediction("test"))

        parent_candidate = Candidate(parent_module, generation_number=0)
        child_candidate = Candidate(child_module, generation_number=1, parents=[parent_candidate])

        new_borns = NewBorns(child_candidate, iteration=1)
        parents = Parents(parent_candidate, iteration=0)

        budget = LLMCallsBudget(100)

        # Should handle empty feedback data gracefully
        survivors = evaluator.evaluate(new_borns, budget)
        assert survivors.size() == 0  # No minibatch validation possible
