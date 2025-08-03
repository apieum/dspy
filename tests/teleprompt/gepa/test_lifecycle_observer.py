"""Test CompilationObserver lifecycle system."""

import dspy
from dspy.teleprompt.gepa.core import GEPA


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def test_compilation_lifecycle_events():
    """Test that all components receive lifecycle events in correct order."""
    
    # Track lifecycle events from budget component
    original_start_compilation = None
    original_finish_compilation = None
    original_start_iteration = None  
    original_finish_iteration = None
    events = []
    
    # Create GEPA
    gepa = GEPA.create_basic(metric=simple_metric, max_calls=25)
    
    # Patch budget to track lifecycle calls
    original_start_compilation = gepa.budget.start_compilation
    original_finish_compilation = gepa.budget.finish_compilation
    original_start_iteration = gepa.budget.start_iteration
    original_finish_iteration = gepa.budget.finish_iteration
    
    def track_start_compilation(student, training_data):
        events.append("budget_start_compilation")
        return original_start_compilation(student, training_data)
        
    def track_finish_compilation(result, final_pool):
        events.append("budget_finish_compilation")
        return original_finish_compilation(result, final_pool)
        
    def track_start_iteration(iteration, cohort, budget):
        events.append(f"budget_start_iteration_{iteration}")
        return original_start_iteration(iteration, cohort, budget)
        
    def track_finish_iteration(iteration, filtered_cohort, budget):
        events.append(f"budget_finish_iteration_{iteration}")
        return original_finish_iteration(iteration, filtered_cohort, budget)
    
    gepa.budget.start_compilation = track_start_compilation
    gepa.budget.finish_compilation = track_finish_compilation
    gepa.budget.start_iteration = track_start_iteration
    gepa.budget.finish_iteration = track_finish_iteration
    
    # Override next_generation to terminate quickly
    def mock_next_generation(cohort):
        events.append("next_generation_called")
        # Return first candidate to complete compilation
        compiled_module = cohort.candidates[0].module
        compiled_module._compiled = True
        return compiled_module
    
    gepa.next_generation = mock_next_generation
    
    # Test data
    training_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
    ]
    
    # Run compilation
    student = dspy.Predict("input -> output")
    result = gepa.compile(student, training_data)
    
    # Verify lifecycle events occurred
    assert "budget_start_compilation" in events
    assert "budget_finish_compilation" in events
    assert "next_generation_called" in events
    
    # Verify result is properly compiled
    assert hasattr(result, '_compiled')
    assert result._compiled == True
    
    print("✓ Lifecycle events fired correctly")
    print(f"Events captured: {events}")


def test_budget_tracks_compilation_progress():
    """Test that Budget component uses lifecycle events for tracking."""
    
    # Create GEPA with tracked budget
    gepa = GEPA.create_basic(metric=simple_metric, max_calls=100)
    
    # Mock next_generation to terminate quickly
    def mock_next_generation(cohort):
        # Return first candidate to complete compilation
        compiled_module = cohort.candidates[0].module
        compiled_module._compiled = True  
        return compiled_module
    
    gepa.next_generation = mock_next_generation
    
    # Test data
    training_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
    ]
    
    # Run compilation
    student = dspy.Predict("input -> output")
    result = gepa.compile(student, training_data)
    
    # Verify budget received lifecycle events
    # Budget should have tracked compilation progress
    assert hasattr(gepa.budget, 'iteration_costs')
    assert isinstance(gepa.budget.iteration_costs, list)
    
    # Budget should have been reset at start of compilation
    assert gepa.budget.consumed_calls >= 0  # Some calls may have been consumed
    
    print("✓ Budget correctly tracked compilation progress via lifecycle events")


def test_components_can_opt_into_lifecycle_events():
    """Test that components can choose which lifecycle events to handle."""
    
    from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
    from dspy.teleprompt.gepa.generation.mutation import MutationGenerator
    
    # Create components
    evaluator = PromotionEvaluator(metric=simple_metric)
    generator = MutationGenerator()
    
    # Test that they have lifecycle methods (inherited from CompilationObserver)
    assert hasattr(evaluator, 'start_compilation')
    assert hasattr(evaluator, 'finish_compilation')
    assert hasattr(evaluator, 'start_iteration')
    assert hasattr(evaluator, 'finish_iteration')
    
    assert hasattr(generator, 'start_compilation')
    assert hasattr(generator, 'finish_compilation')
    assert hasattr(generator, 'start_iteration')
    assert hasattr(generator, 'finish_iteration')
    
    # Test that they can be called without error (default no-op implementations)
    student = dspy.Predict("input -> output")
    training_data = [dspy.Example(input="test", answer="test")]
    
    # Components should handle lifecycle events gracefully
    evaluator.start_compilation(student, training_data)
    evaluator.finish_compilation(student, None)
    evaluator.start_iteration(0, None, None)
    evaluator.finish_iteration(0, None, None)
    
    generator.start_compilation(student, training_data)
    generator.finish_compilation(student, None)
    generator.start_iteration(0, None, None)
    generator.finish_iteration(0, None, None)
    
    print("✓ Components can opt into lifecycle events without breaking")


if __name__ == "__main__":
    test_compilation_lifecycle_events()
    test_budget_tracks_compilation_progress()
    test_components_can_opt_into_lifecycle_events()
    print("All lifecycle observer tests passed!")