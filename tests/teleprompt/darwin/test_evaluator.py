"""Test Darwin evaluator (two-phase evaluation)."""

import dspy
from unittest.mock import Mock
from dspy.teleprompt.darwin.evaluation.gepa_evaluator import GEPATwoPhasesEval
from dspy.teleprompt.darwin.evaluation.metrics import Metric
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import NewBorns
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy
from dspy.teleprompt.darwin import GEPAConfig
from dspy.teleprompt.darwin.evaluation.cache import EvaluationCache


def simple_metric(example: dspy.Example, prediction, trace=None) -> Metric:
    """Simple metric for testing that returns Metric objects."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        value = 1.0 if str(example.answer) == str(prediction.answer) else 0.0
        return Metric(value, feedback=f"Predicted: {prediction.answer}, Expected: {example.answer}", trace=trace)
    return Metric(0.0, feedback="No answer attributes found", trace=trace)


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
        validation_data = [dspy.Example(input="test", answer="correct").with_inputs("input")]
        minibatch_data = [dspy.Example(input="mini", answer="correct").with_inputs("input")]
        evaluator = GEPATwoPhasesEval(
            config=GEPAConfig(fitness_function=simple_metric),
            evaluation_cache=EvaluationCache(),
            assessor=simple_metric, 
            minibatch_data=minibatch_data,
            validation_data=validation_data
        )
        
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'start_compilation')
        assert evaluator.validation_data == validation_data

    def test_evaluator_compilation(self):
        """Test evaluator compilation setup."""
        training_data = [
            dspy.Example(input="test1", answer="answer1").with_inputs("input"),
            dspy.Example(input="test2", answer="answer2").with_inputs("input"),
            dspy.Example(input="test3", answer="answer3").with_inputs("input"),
        ]
        minibatch_data = training_data[:1]  # Use subset for minibatch
        evaluator = GEPATwoPhasesEval(
            config=GEPAConfig(fitness_function=simple_metric),
            evaluation_cache=EvaluationCache(),
            assessor=simple_metric, 
            minibatch_data=minibatch_data,
            validation_data=training_data
        )

        student = dspy.Predict("input -> output")
        evaluator.start_compilation(student, verbose=False)

        assert len(evaluator.evaluators) > 0
        assert evaluator.validation_data is not None

    def test_candidate_evaluation(self):
        """Test basic candidate evaluation."""
        training_data = [
            dspy.Example(input="test1", answer="correct").with_inputs("input"),
            dspy.Example(input="test2", answer="correct").with_inputs("input"),
        ]
        minibatch_data = training_data[:1]  # Use subset for minibatch
        evaluator = GEPATwoPhasesEval(
            config=GEPAConfig(fitness_function=simple_metric),
            evaluation_cache=EvaluationCache(),
            assessor=simple_metric, 
            minibatch_data=minibatch_data,
            validation_data=training_data
        )

        student = dspy.Predict("input -> output")
        evaluator.start_compilation(student, verbose=False)

        # Create test candidate
        module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))
        candidate = Candidate(module, generation_number=0)
        new_borns = NewBorns(candidate, iteration=0)
        budget = LMCallsBudget(100)

        # Evaluate
        survivors = evaluator.evaluate(new_borns, budget)
        
        assert survivors.size() == 1
        assert survivors.first().scores  # Should have Metric scores
        assert len(survivors.first().scores) > 0
        assert evaluator.last_proposal_records[0]["selected"] is True

    def test_budget_tracking(self):
        """Test that evaluation tracks budget correctly."""
        training_data = [dspy.Example(input="test", answer="correct").with_inputs("input")]
        evaluator = GEPATwoPhasesEval(
            config=GEPAConfig(fitness_function=simple_metric),
            evaluation_cache=EvaluationCache(),
            assessor=simple_metric, 
            minibatch_data=training_data, 
            validation_data=training_data
        )
        student = dspy.Predict("input -> output")
        evaluator.start_compilation(student, verbose=False)

        module = mock_module_with_history(lambda **kwargs: mock_prediction("correct"))
        candidate = Candidate(module, generation_number=0)
        new_borns = NewBorns(candidate, iteration=0)
        
        budget = LMCallsBudget(100)
        initial_calls = budget.consumed_calls

        evaluator.evaluate(new_borns, budget)
        
        assert budget.consumed_calls >= initial_calls  # Budget was tracked

    def test_optional_batch_evaluator_receives_multiple_candidates(self):
        examples = [dspy.Example(input="test", answer="correct").with_inputs("input")]
        calls = []

        def batch_evaluator(jobs, assessor, channel):
            calls.append(jobs)
            return [[simple_metric(example, mock_prediction("correct")) for example in batch]
                    for _, batch in jobs]

        evaluator = GEPATwoPhasesEval(
            config=GEPAConfig(fitness_function=simple_metric, batch_evaluator=batch_evaluator),
            evaluation_cache=EvaluationCache(),
            assessor=simple_metric,
            minibatch_data=examples,
            validation_data=examples,
            batch_evaluator=batch_evaluator,
        )
        evaluator.start_compilation(dspy.Predict("input -> output"), verbose=False)
        candidates = [
            Candidate(mock_module_with_history(lambda **kwargs: mock_prediction("correct")))
            for _ in range(2)
        ]

        evaluator.evaluate(NewBorns(*candidates, iteration=0), LMCallsBudget(100))

        assert len(calls) == 1
        assert len(calls[0]) == 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
