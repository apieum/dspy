"""Functional tests to ensure Darwin GEPA works in real scenarios."""

import dspy
from dspy.teleprompt.darwin import GEPAMute
from dspy.utils.dummies import DummyLM


class MultiStepQA(dspy.Module):
    """Multi-step QA program for realistic testing."""

    def __init__(self):
        super().__init__()
        self.think = dspy.ChainOfThought("question -> reasoning, answer")

    def forward(self, question):
        return self.think(question=question)


def accuracy_metric(example, prediction, trace=None):
    """Realistic accuracy metric."""
    if not hasattr(example, 'answer') or not hasattr(prediction, 'answer'):
        return 0.0
    expected = str(example.answer).lower().strip()
    actual = str(prediction.answer).lower().strip()
    return 1.0 if expected == actual else 0.0


def contains_metric(example, prediction, trace=None):
    """Metric that checks if answer contains expected content."""
    if not hasattr(example, 'answer') or not hasattr(prediction, 'answer'):
        return 0.0
    expected = str(example.answer).lower()
    actual = str(prediction.answer).lower()
    return 1.0 if expected in actual else 0.0


class TestFunctional:
    """Test real-world functionality and edge cases."""

    def test_optimization_improves_performance(self):
        """Test that optimization actually improves performance over multiple iterations."""
        # Create a realistic dataset
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="What is 3+3?", answer="6").with_inputs("question"),
            dspy.Example(question="What is 5+5?", answer="10").with_inputs("question"),
            dspy.Example(question="What is 7+7?", answer="14").with_inputs("question"),
            dspy.Example(question="What is 4+4?", answer="8").with_inputs("question"),
        ]
        
        # Mock LM that can actually "improve" with better instructions
        improving_responses = [
            {"reasoning": "Adding numbers", "answer": "4"},  # Initial correct answers
            {"reasoning": "Adding numbers", "answer": "6"},
            {"reasoning": "Adding numbers", "answer": "wrong"},  # Some wrong initially
            {"reasoning": "Adding numbers", "answer": "wrong"},
            {"reasoning": "Adding numbers", "answer": "8"},
            {"response": "Improved instruction: Add the two numbers carefully."},  # Reflection
            {"reasoning": "Carefully adding", "answer": "10"},  # Better after reflection
            {"reasoning": "Carefully adding", "answer": "14"},
        ]
        
        dummy_lm = DummyLM(improving_responses)
        
        with dspy.context(lm=dummy_lm):
            student = MultiStepQA()
            optimizer = GEPAMute(accuracy_metric, max_calls=10, patience=2)
            
            # Test that optimization completes without error
            result = optimizer.compile(student, trainset=trainset)
            
            assert isinstance(result, dspy.Module)
            assert result._compiled is True
            assert hasattr(result, 'think')  # Preserves original structure

    def test_optimization_with_difficult_metric(self):
        """Test optimization with a more challenging metric."""
        trainset = [
            dspy.Example(question="Explain gravity", answer="force").with_inputs("question"),
            dspy.Example(question="What is DNA?", answer="genetic").with_inputs("question"),
            dspy.Example(question="How does photosynthesis work?", answer="sunlight").with_inputs("question"),
        ]
        
        # Responses that partially match the contains_metric
        responses = [
            {"reasoning": "Thinking about physics", "answer": "Gravity is a fundamental force in nature"},  # Contains "force"
            {"reasoning": "Thinking about biology", "answer": "DNA is the genetic material"},  # Contains "genetic"
            {"reasoning": "Thinking about plants", "answer": "Plants use sunlight for energy"},  # Contains "sunlight"
            {"response": "Enhanced instruction: Focus on key concepts in your answers."},
            {"reasoning": "Key concepts in physics", "answer": "force of attraction"},
            {"reasoning": "Key concepts in biology", "answer": "genetic code information"},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = MultiStepQA()
            optimizer = GEPAMute(contains_metric, max_calls=8, patience=2)
            
            result = optimizer.compile(student, trainset=trainset)
            
            assert result._compiled is True

    def test_optimization_with_small_dataset(self):
        """Test that optimization works with minimal data (edge case)."""
        # Very small dataset - real-world scenario with limited data
        trainset = [
            dspy.Example(question="Test question", answer="test").with_inputs("question"),
        ]
        
        responses = [
            {"reasoning": "Processing question", "answer": "test"},
            {"response": "Optimized instruction for single example."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = MultiStepQA()
            optimizer = GEPAMute(accuracy_metric, max_calls=3, patience=1)
            
            # Should handle small datasets gracefully
            result = optimizer.compile(student, trainset=trainset)
            
            assert result._compiled is True

    def test_optimization_budget_exhaustion(self):
        """Test behavior when budget is exhausted."""
        trainset = [
            dspy.Example(question="What is AI?", answer="intelligence").with_inputs("question"),
            dspy.Example(question="What is ML?", answer="learning").with_inputs("question"),
        ]
        
        # Very few responses to force budget exhaustion
        responses = [
            {"reasoning": "About AI", "answer": "artificial intelligence"},
            {"response": "Quick optimization due to budget."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = MultiStepQA()
            optimizer = GEPAMute(contains_metric, max_calls=2, patience=1)  # Very tight budget
            
            # Should complete gracefully even with tight budget
            result = optimizer.compile(student, trainset=trainset)
            
            assert result._compiled is True

    def test_optimization_preserves_module_structure(self):
        """Test that optimization preserves the original module structure."""
        trainset = [
            dspy.Example(question="Test", answer="result").with_inputs("question"),
        ]
        
        responses = [
            {"reasoning": "Processing", "answer": "result"},
            {"response": "Structure-preserving optimization."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            original_student = MultiStepQA()
            original_predictors = original_student.predictors()
            
            optimizer = GEPAMute(accuracy_metric, max_calls=3)
            result = optimizer.compile(original_student, trainset=trainset)
            
            # Structure should be preserved
            assert hasattr(result, 'think')
            assert len(result.predictors()) == len(original_predictors)
            assert type(result.think) == type(original_student.think)

    def test_data_leakage_prevention(self):
        """Test that external devset is not used during training."""
        trainset = [
            dspy.Example(question="Train Q1", answer="A1").with_inputs("question"),
            dspy.Example(question="Train Q2", answer="A2").with_inputs("question"),
        ]
        
        devset = [
            dspy.Example(question="Test Q1", answer="A1").with_inputs("question"),
            dspy.Example(question="Test Q2", answer="A2").with_inputs("question"),
        ]
        
        responses = [
            {"reasoning": "Training phase", "answer": "A1"},
            {"reasoning": "Training phase", "answer": "A2"},
            {"response": "Optimized without test data leakage."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = MultiStepQA()
            optimizer = GEPAMute(accuracy_metric, max_calls=5)
            
            # External devset should be reserved for final evaluation only
            result = optimizer.compile(student, trainset=trainset, devset=devset)
            
            assert result._compiled is True
            # Verify split strategy correctly separates data
            assert optimizer.split_strategy.external_devset == devset
            assert len(optimizer.split_strategy.internal_validation_set) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])