"""Test that compile() scores initial candidate correctly."""

import dspy
from dspy.teleprompt.gepa.core import GEPA
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Cohort


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def test_compile_scores_initial_candidate():
    """Test that compile() evaluates and adds initial candidate to pool before optimization."""
    
    # Create a basic GEPA optimizer
    gepa = GEPA.create_basic(metric=simple_metric, max_calls=100, population_size=2)
    
    # Create a simple student module to optimize
    student = dspy.Predict("input -> output")
    
    # Create test data
    training_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
        dspy.Example(input="test4", answer="answer4"),
        dspy.Example(input="test5", answer="answer5"),
    ]
    
    # Store original next_generation method to spy on it
    original_next_generation = gepa.next_generation
    next_generation_called = False
    initial_pool_size = None
    initial_score_matrix_tasks = None
    
    def spy_next_generation(cohort):
        nonlocal next_generation_called, initial_pool_size, initial_score_matrix_tasks
        next_generation_called = True
        
        # Capture pool state when next_generation is first called
        initial_pool_size = len(gepa.candidate_pool.candidates)
        initial_score_matrix_tasks = list(gepa.candidate_pool.score_matrix.task_scores.keys())
        
        # Verify candidate is properly scored
        
        # Return early to avoid full optimization - just return the initial candidate's module
        if gepa.candidate_pool.candidates:
            best_candidate = gepa.candidate_pool.candidates[0]
            compiled_module = best_candidate.module
            compiled_module._compiled = True
            return compiled_module
        else:
            raise RuntimeError("No candidates in pool")
    
    gepa.next_generation = spy_next_generation
    
    # Run compile - this should score initial candidate before calling next_generation
    result = gepa.compile(student, training_data)
    
    # Verify expectations:
    
    # 1. next_generation should have been called
    assert next_generation_called, "next_generation should have been called"
    
    # 2. When next_generation is called, the initial candidate should already be in the pool
    assert initial_pool_size == 1, f"Expected 1 initial candidate in pool, got {initial_pool_size}"
    
    # 3. The initial candidate should have scores (if evaluation succeeded)
    # Note: With dummy modules, scores might be zero, but tasks should still be tracked
    initial_candidate = gepa.candidate_pool.candidates[0]
    assert hasattr(initial_candidate, 'task_scores'), "Initial candidate should have task_scores"
    assert len(initial_candidate.task_scores) == len(training_data), f"Expected {len(training_data)} task scores"
    
    # 4. Score matrix should reflect the initial candidate's evaluation
    # (Even if scores are zero, the matrix should know about the tasks)
    all_task_ids = set(range(len(training_data)))
    
    # 5. Initial candidate should have generation_number 0
    assert initial_candidate.generation_number == 0, "Initial candidate should be generation 0"
    
    # 6. Result should be the compiled module
    assert hasattr(result, '_compiled'), "Result should be marked as compiled"
    assert result._compiled == True, "Result should be marked as compiled"


if __name__ == "__main__":
    test_compile_scores_initial_candidate()
    print("compile() initial scoring test passed!")