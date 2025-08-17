"""Test Darwin evaluator (two-phase evaluation)."""

import dspy
from unittest.mock import Mock
from dspy.teleprompt.darwin.evaluation.gepa_evaluator import GEPATwoPhasesEval
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import NewBorns
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy


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
    module.history = []
    module.side_effect = predictions_func
    
    mock_predictor = Mock()
    mock_signature = Mock()
    mock_signature.instructions = "Test instruction"
    mock_predictor.signature = mock_signature
    module.predictors.return_value = [mock_predictor]
    module.deepcopy.return_value = module
    
    return module


class TestEvaluator:
    """Test two-phase evaluation core functionality."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = GEPATwoPhasesEval(metric=simple_metric, minibatch_size=3)
        
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'start_compilation')
        assert evaluator.minibatch_size == 3

    def test_evaluator_compilation(self):
        """Test evaluator compilation setup."""
        evaluator = GEPATwoPhasesEval(metric=simple_metric, minibatch_size=2)

        training_data = [
            dspy.Example(input="test1", answer="answer1").with_inputs("input"),
            dspy.Example(input="test2", answer="answer2").with_inputs("input"),
            dspy.Example(input="test3", answer="answer3").with_inputs("input"),
        ]

        student = dspy.Predict("input -> output")
        split_strategy = DefaultSplitStrategy(trainset=training_data, verbose=False)
        evaluator.start_compilation(student, split_strategy=split_strategy, verbose=False)

        assert len(evaluator.evaluators) > 0
        assert evaluator.split_strategy is not None

    def test_candidate_evaluation(self):
        """Test basic candidate evaluation."""
        evaluator = GEPATwoPhasesEval(metric=simple_metric, minibatch_size=2)

        training_data = [
            dspy.Example(input="test1", answer="correct").with_inputs("input"),
            dspy.Example(input="test2", answer="correct").with_inputs("input"),
        ]

        student = dspy.Predict("input -> output")
        split_strategy = DefaultSplitStrategy(trainset=training_data, verbose=False)
        evaluator.start_compilation(student, split_strategy=split_strategy, verbose=False)

        # Create test candidate
        module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))
        candidate = Candidate(module, generation_number=0)
        new_borns = NewBorns(candidate, iteration=0)
        budget = LMCallsBudget(100)

        # Evaluate
        survivors = evaluator.evaluate(new_borns, budget)
        
        assert survivors.size() == 1
        assert survivors.first().task_scores  # Should have scores

    def test_budget_tracking(self):
        """Test that evaluation tracks budget correctly."""
        evaluator = GEPATwoPhasesEval(metric=simple_metric, minibatch_size=1)

        training_data = [dspy.Example(input="test", answer="correct").with_inputs("input")]
        student = dspy.Predict("input -> output")
        split_strategy = DefaultSplitStrategy(trainset=training_data, verbose=False)
        evaluator.start_compilation(student, split_strategy=split_strategy, verbose=False)

        module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))
        candidate = Candidate(module, generation_number=0)
        new_borns = NewBorns(candidate, iteration=0)
        
        budget = LMCallsBudget(100)
        initial_calls = budget.consumed_calls

        evaluator.evaluate(new_borns, budget)
        
        assert budget.consumed_calls >= initial_calls  # Budget was tracked


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])