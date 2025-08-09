"""Test clean separation of concerns in the new GEPA design."""

import dspy
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import NewBorns, Survivors
from dspy.teleprompt.gepa.selection import ParetoFrontier
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget
from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManager
from unittest.mock import Mock


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def test_evaluator_handles_two_phase_evaluation():
    """Test that evaluator handles two-phase evaluation cleanly."""

    # Setup test data with proper inputs
    training_data = [
        dspy.Example(input="feedback1", answer="answer1").with_inputs("input"),
        dspy.Example(input="feedback2", answer="answer2").with_inputs("input"),
        dspy.Example(input="pareto1", answer="answer1").with_inputs("input"),
    ]

    # Create evaluator with DatasetManager
    evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)
    student = dspy.Predict("input -> output")
    dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.33)
    evaluator.start_compilation(student, dataset_manager)

    # Create test candidates (parentless - should be evaluated directly)
    from unittest.mock import Mock
    module1 = Mock()
    module1.history = []
    module1.return_value = Mock(answer="answer1")
    candidate1 = Candidate(module1, generation_number=0)  # No parents

    # Test direct evaluation (no parent comparison)
    new_borns = NewBorns(candidate1, iteration=0)
    budget = LLMCallsBudget(100)

    survivors = evaluator.evaluate(new_borns, budget)

    # Verify clean interface
    assert isinstance(survivors, Survivors)
    assert survivors.size() == 1
    assert survivors.first().task_scores  # Should have scores populated


def test_selector_manages_population_internally():
    """Test that selector manages its population without external interference."""

    # Setup test data
    training_data = [
        dspy.Example(input="f1", answer="a1"),
        dspy.Example(input="p1", answer="a1")
    ]

    # Create selector with DatasetManager
    selector = ParetoFrontier()
    student = dspy.Predict("input -> output")
    dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.5)
    selector.start_compilation(student, dataset_manager)

    # Create test candidates with scores
    candidate1 = Candidate(Mock(), generation_number=0)
    candidate1.task_scores = {0: 0.8}

    candidate2 = Candidate(Mock(), generation_number=0)
    candidate2.task_scores = {0: 0.9}

    # Test promotion
    survivors = Survivors(candidate1, candidate2, iteration=0)
    parents = selector.promote(survivors)

    # Verify selector manages population internally
    assert parents.size() >= 1  # At least some candidates promoted
    assert selector.size() > 0  # Selector tracks population internally


if __name__ == "__main__":
    test_evaluator_handles_two_phase_evaluation()
    test_selector_manages_population_internally()
    print("Clean separation tests passed!")
