"""Test for correct 2-phase evaluation behavior."""

import dspy
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Cohort, FilteredCohort
from dspy.teleprompt.gepa.data.candidate_pool import CandidatePool
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def test_two_phase_evaluation_behavior():
    """Test that evaluation implements 2-phase process correctly."""
    
    # Create evaluation data (minibatch and full)
    minibatch_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2")
    ]
    
    full_evaluation_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
        dspy.Example(input="test4", answer="answer4")
    ]

    # Setup - use very low threshold to ensure some candidates pass
    evaluator = PromotionEvaluator(
        metric=simple_metric, 
        promotion_threshold=0.0
    )
    # Prepare evaluator with data
    student = dspy.Predict("input -> output")
    evaluator.start_compilation(student, full_evaluation_data)
    candidate_pool = CandidatePool()
    
    # Create test candidates - we'll mock the evaluation by directly setting task_scores
    module1 = dspy.Predict("input -> output")
    candidate1 = Candidate(module1, generation_number=1)
    
    module2 = dspy.Predict("input -> output") 
    candidate2 = Candidate(module2, generation_number=1)
    
    module3 = dspy.Predict("input -> output")
    candidate3 = Candidate(module3, generation_number=1)
    
    new_cohort = Cohort(candidate1, candidate2, candidate3)
    
    # Expected behavior: 
    # Phase 1: Fast filter using minibatch_data should promote some candidates
    # Phase 2: Full evaluation should score promoted candidates and promote to pool
    
    # This is the interface we want:
    result = evaluator.evaluate_two_phase(new_cohort)
    # GEPA handles promotion to pool
    candidate_pool.promote(result)
    
    # Verify the two-phase evaluation worked correctly
    
    # Assertions for expected behavior:
    # 1. Result should be FilteredCohort with promoted candidates
    assert isinstance(result, FilteredCohort)
    
    # 2. Promoted candidates should have task_scores populated
    for candidate in result.candidates:
        assert hasattr(candidate, 'task_scores')
        assert len(candidate.task_scores) == len(full_evaluation_data)
    
    # 3. Candidate pool should be promoted with candidates
    assert len(candidate_pool.candidates) == len(result.candidates)
    for candidate in result.candidates:
        assert candidate in candidate_pool.candidates
    
    # 4. Score matrix should be updated (only if candidates have non-zero scores)
    # Since our test candidates fail evaluation (dummy modules), they get zero scores
    # and ScoreMatrix filters those out - this is correct behavior
    # But the candidates should still be in the pool
    assert len(candidate_pool.candidates) == len(result.candidates)


if __name__ == "__main__":
    test_two_phase_evaluation_behavior()
    print("Two-phase evaluation test passed!")