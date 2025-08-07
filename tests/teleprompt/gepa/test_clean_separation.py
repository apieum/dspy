"""Test that evaluator and GEPA have clean separation of concerns."""

import dspy
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Cohort
from dspy.teleprompt.gepa.selection import ParetoFrontier
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def test_evaluator_returns_cohorts_gepa_manages_pool():
    """Test that evaluator returns cohorts and GEPA manages candidate pool."""

    # Setup data
    evaluation_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
    ]

    # Create evaluator (should not know about candidate pool)
    evaluator = PromotionEvaluator(
        metric=simple_metric,
        promotion_threshold=0.0  # Accept all candidates
    )
    # Prepare evaluator with data
    student = dspy.Predict("input -> output")
    d_feedback = evaluation_data
    d_pareto = evaluation_data
    evaluator.start_compilation(student, d_feedback, d_pareto)

    # Create selector (managed by GEPA)
    selector = ParetoFrontier()
    selector.start_compilation(student, d_feedback, d_pareto)

    # Create test candidates
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=1)

    module2 = dspy.Predict("input -> output")
    candidate2 = Candidate(module2, generation_number=1)

    test_cohort = Cohort(candidate1, candidate2)

    # PHASE 1: Evaluator evaluates and returns cohort (no pool interaction)
    budget = LLMCallsBudget(100)  # Mock budget for testing
    evaluated_cohort = evaluator.evaluate_for_promotion(test_cohort, budget)

    # Verify evaluator populated task_scores but didn't touch selector
    assert evaluated_cohort.size() == 2
    assert selector.size() == 0  # Selector should still be empty

    for candidate in evaluated_cohort:
        assert hasattr(candidate, 'task_scores')
        assert len(candidate.task_scores) == len(evaluation_data)

    # PHASE 2: GEPA handles selector management
    selector.promote(evaluated_cohort)

    # Verify GEPA successfully promoted candidates to selector
    assert selector.size() == 2

    # Verify candidates were tracked in selector
    assert selector.size() == 2


def test_evaluation_separation():
    """Test evaluation maintains clean separation."""

    # Setup data
    evaluation_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
        dspy.Example(input="test4", answer="answer4")
    ]

    # Create evaluator
    evaluator = PromotionEvaluator(
        metric=simple_metric,
        promotion_threshold=0.0
    )
    # Prepare evaluator with data
    student = dspy.Predict("input -> output")
    d_feedback = evaluation_data
    d_pareto = evaluation_data
    evaluator.start_compilation(student, d_feedback, d_pareto)

    # Create selector
    selector = ParetoFrontier()
    selector.start_compilation(student, d_feedback, d_pareto)

    # Create test candidates
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=1)

    module2 = dspy.Predict("input -> output")
    candidate2 = Candidate(module2, generation_number=1)

    new_cohort = Cohort(candidate1, candidate2)

    # PHASE 1: Evaluator performs evaluation (no pool interaction)
    budget = LLMCallsBudget(100)  # Mock budget for testing
    filtered_cohort = evaluator.evaluate(new_cohort, budget)

    # Verify evaluator did its job but didn't touch selector
    assert selector.size() == 0  # Selector should still be empty
    assert filtered_cohort.size() >= 0  # Some candidates might be filtered

    # PHASE 2: GEPA handles pool promotion using evaluate_for_promotion
    evaluated_cohort = evaluator.evaluate_for_promotion(new_cohort, budget)

    # Verify task_scores are populated
    for candidate in evaluated_cohort:
        assert hasattr(candidate, 'task_scores')
        assert len(candidate.task_scores) == len(evaluation_data)

    # PHASE 3: GEPA handles selector promotion
    selector.promote(evaluated_cohort)

    # Verify GEPA successfully promoted candidates
    assert selector.size() == evaluated_cohort.size()


if __name__ == "__main__":
    test_evaluator_returns_cohorts_gepa_manages_pool()
    test_evaluation_separation()
    print("Clean separation tests passed!")
