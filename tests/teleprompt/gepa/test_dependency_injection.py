"""Test proper dependency injection design for GEPA components."""

import dspy
from dspy.teleprompt.gepa.core import GEPA
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
from dspy.teleprompt.gepa.generation import ReflectivePromptMutation
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def test_components_configure_themselves():
    """Test that components handle their own configuration with start_compilation()."""
    
    # Create training data
    training_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
        dspy.Example(input="test4", answer="answer4"),
        dspy.Example(input="test5", answer="answer5"),
    ]
    
    # Create components without any dataset knowledge
    evaluator = PromotionEvaluator(metric=simple_metric, promotion_threshold=0.3)
    generator = ReflectivePromptMutation()
    
    # Verify components start without dataset knowledge
    assert evaluator.evaluation_data == []
    assert evaluator.minibatch_data == []
    assert generator.feedback_data == []
    
    # Components configure themselves when given training data
    student = dspy.Predict("input -> output")
    evaluator.start_compilation(student, training_data)
    generator.start_compilation(student, training_data)
    
    # Verify components now have dataset knowledge
    assert evaluator.evaluation_data == training_data
    assert len(evaluator.minibatch_data) == 1  # 20% of 5 = 1
    assert evaluator.minibatch_data == training_data[:1]
    assert generator.feedback_data == training_data
    
    print("✓ Components successfully configured themselves")


def test_gepa_does_not_configure_components():
    """Test that GEPA delegates configuration to components via start_compilation()."""
    
    # Create training data
    training_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
    ]
    
    # Create GEPA optimizer - components should be unconfigured
    gepa = GEPA.create_basic(metric=simple_metric, max_calls=50, population_size=2)
    
    # Verify components start unconfigured
    assert gepa.evaluator.evaluation_data == []
    assert gepa.generator.feedback_data == []
    
    # Create a simple student module
    student = dspy.Predict("input -> output")
    
    # Mock next_generation to verify preparation happened before optimization
    def mock_next_generation(cohort):
        # When next_generation is called, components should be configured
        assert gepa.evaluator.evaluation_data == training_data
        assert gepa.generator.feedback_data == training_data
        
        # Return a compiled module to complete the test
        compiled_module = cohort.first().module
        compiled_module._compiled = True
        return compiled_module
    
    gepa.next_generation = mock_next_generation
    
    # Compile should trigger component preparation
    result = gepa.compile(student, training_data)
    
    # Verify result is properly compiled
    assert hasattr(result, '_compiled')
    assert result._compiled == True
    
    print("✓ GEPA successfully delegated configuration to components")


def test_single_training_dataset_design():
    """Test that GEPA uses single training dataset (no artificial splits)."""
    
    training_data = [
        dspy.Example(input="test1", answer="answer1"),
        dspy.Example(input="test2", answer="answer2"),
        dspy.Example(input="test3", answer="answer3"),
        dspy.Example(input="test4", answer="answer4"),
    ]
    
    # Create GEPA optimizer
    gepa = GEPA.create_basic(metric=simple_metric, max_calls=30)
    student = dspy.Predict("input -> output")
    
    # Mock next_generation to verify single dataset usage
    def mock_next_generation(cohort):
        # Both evaluator and generator should use the same training data
        assert gepa.evaluator.evaluation_data == training_data
        assert gepa.generator.feedback_data == training_data
        
        # Each component can create its own minibatches/subsets as needed
        # Evaluator creates minibatch for 2-phase evaluation
        assert len(gepa.evaluator.minibatch_data) == 1  # 20% of 4 = 1
        assert gepa.evaluator.minibatch_data == training_data[:1]
        
        # Generator uses full dataset for feedback
        assert gepa.generator.feedback_data == training_data
        
        # Return compiled module
        compiled_module = cohort.first().module  
        compiled_module._compiled = True
        return compiled_module
    
    gepa.next_generation = mock_next_generation
    
    # Compile with single training dataset (no artificial pareto_data/feedback_data split)
    result = gepa.compile(student, training_data)
    
    assert result._compiled == True
    
    print("✓ GEPA correctly uses single training dataset")


if __name__ == "__main__":
    test_components_configure_themselves()
    test_gepa_does_not_configure_components() 
    test_single_training_dataset_design()
    print("All dependency injection tests passed!")