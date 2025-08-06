"""Test GEPA core algorithm and compilation flow."""

import dspy
from dspy.teleprompt.gepa.core import GEPA


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


class TestGEPACore:
    """Test GEPA core algorithm behavior."""
    
    def test_gepa_returns_compiled_program(self):
        """Test that GEPA returns a compiled program."""
        training_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
        ]
        
        gepa = GEPA.create_basic(metric=simple_metric, max_calls=25)
        student = dspy.Predict("input -> output")
        
        # Mock next_generation to terminate quickly
        def mock_next_generation(cohort):
            compiled_module = cohort.first().module
            compiled_module._compiled = True
            return compiled_module
        
        gepa.next_generation = mock_next_generation
        
        result = gepa.compile(student, training_data)
        
        assert hasattr(result, '_compiled')
        assert result._compiled == True
    
    def test_gepa_algorithm_phases(self):
        """Test that GEPA follows the expected algorithm phases."""
        training_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
        ]
        
        gepa = GEPA.create_basic(metric=simple_metric, max_calls=50)
        student = dspy.Predict("input -> output")
        
        phases_executed = []
        
        def mock_next_generation(cohort):
            phases_executed.append("next_generation")
            compiled_module = cohort.first().module
            compiled_module._compiled = True
            return compiled_module
        
        gepa.next_generation = mock_next_generation
        
        result = gepa.compile(student, training_data)
        
        # Verify phases were executed
        assert "next_generation" in phases_executed
        assert result._compiled == True
    
    def test_create_basic_gepa(self):
        """Test basic GEPA factory function."""
        gepa = GEPA.create_basic(metric=simple_metric, max_calls=100)
        
        # Verify components are properly initialized
        assert gepa.budget is not None
        assert gepa.evaluator is not None
        assert gepa.generator is not None
        assert hasattr(gepa.budget, 'max_calls')
        assert gepa.budget.max_calls == 100
    
    def test_basic_optimization_workflow(self):
        """Test basic optimization workflow with minimal data."""
        training_data = [
            dspy.Example(input="test", answer="answer"),
        ]
        
        gepa = GEPA.create_basic(metric=simple_metric, max_calls=30)
        student = dspy.Predict("input -> output")
        
        # Mock for quick termination
        def mock_next_generation(cohort):
            compiled_module = cohort.first().module
            compiled_module._compiled = True
            return compiled_module
        
        gepa.next_generation = mock_next_generation
        
        result = gepa.compile(student, training_data)
        
        # Verify workflow completed successfully
        assert result is not None
        assert hasattr(result, '_compiled')
        assert result._compiled == True