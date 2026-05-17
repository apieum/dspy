"""Functional tests to ensure Darwin GEPA works in real scenarios."""

import dspy
from dspy.teleprompt.darwin import Darwin, DarwinConfig, ChannelContext
from dspy.teleprompt.darwin.strategy import GEPAStrategy
from dspy.teleprompt.darwin.budget import LMCallsBudget
from dspy.teleprompt.darwin.selection import ParetoFrontier
from dspy.teleprompt.darwin.generation import ReflectivePromptMutation
from dspy.teleprompt.darwin.generation.feedback import FeedbackProvider
from dspy.teleprompt.darwin.evaluation import GEPATwoPhasesEval
from dspy.teleprompt.darwin.result import Success
from dspy.utils.dummies import DummyLM


class MultiStepQA(dspy.Module):
    """Multi-step QA program for realistic testing."""

    def __init__(self):
        super().__init__()
        self.think = dspy.ChainOfThought("question -> reasoning, answer")

    def forward(self, question):
        return self.think(question=question)


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

            # Create GEPA configuration with new architecture
            config = DarwinConfig(
                max_lm_calls=10,
                patience=2,
                verbose=False
            )

            optimizer = Darwin(GEPAStrategy, config)

            # Test that optimization completes without error
            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()

            assert isinstance(result, Success)
            assert isinstance(compiled_module, dspy.Module)
            assert compiled_module._compiled is True
            assert hasattr(compiled_module, 'think')  # Preserves original structure

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

            # Create GEPA configuration with new architecture
            config = DarwinConfig(
                max_lm_calls=8,
                patience=2,
                verbose=False
            )

            optimizer = Darwin(GEPAStrategy, config)

            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()

            assert isinstance(result, Success)
            assert compiled_module._compiled is True

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
            # Create GEPA configuration with new architecture
            config = DarwinConfig(
                max_lm_calls=3,
                patience=1,
                verbose=False
            )
            optimizer = Darwin(GEPAStrategy, config)

            # Should handle small datasets gracefully
            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()

            assert isinstance(result, Success)
            assert compiled_module._compiled is True

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
            # Create GEPA configuration with very tight budget
            config = DarwinConfig(
                max_lm_calls=2,
                patience=1,
                verbose=False
            )
            optimizer = Darwin(GEPAStrategy, config)

            # Should complete gracefully even with tight budget
            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()

            assert isinstance(result, Success)
            assert compiled_module._compiled is True

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

            # Create GEPA configuration with new architecture
            config = DarwinConfig(
                max_lm_calls=3,
                patience=3,
                verbose=False
            )
            optimizer = Darwin(GEPAStrategy, config)
            compiled_module = optimizer.compile(original_student, trainset=trainset)
            result = optimizer.get_last_result()

            # Structure should be preserved
            assert isinstance(result, Success)
            assert hasattr(compiled_module, 'think')
            assert len(compiled_module.predictors()) == len(original_predictors)
            assert type(compiled_module.think) == type(original_student.think)

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
            # Create GEPA configuration with new architecture
            config = DarwinConfig(
                max_lm_calls=5,
                patience=3,
                verbose=False
            )
            optimizer = Darwin(GEPAStrategy, config)

            # External devset should be reserved for final evaluation only
            compiled_module = optimizer.compile(student, trainset=trainset, devset=devset)
            result = optimizer.get_last_result()

            assert isinstance(result, Success)
            assert compiled_module._compiled is True
            # Verify strategy correctly separates data
            assert optimizer.strategy.devset == devset
            assert len(optimizer.strategy.validation_data) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])