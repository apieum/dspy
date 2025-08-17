"""Integration tests for Darwin GEPA optimization."""

import dspy
from dspy.teleprompt.darwin import GEPAMute, GEPAMerge
from dspy.utils.dummies import DummyLM


class SimpleQA(dspy.Module):
    """Simple QA program for testing."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


def simple_metric(example, prediction, trace=None):
    """Simple metric for testing."""
    expected = example.answer.lower() if hasattr(example, 'answer') else ""
    actual = prediction.answer.lower() if hasattr(prediction, 'answer') else ""
    return 1.0 if expected == actual else 0.0


class TestIntegration:
    """Test end-to-end GEPA functionality."""

    def test_gepa_mute_compilation(self):
        """Test GEPAMute end-to-end compilation."""
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="What color is the sky?", answer="blue").with_inputs("question"),
            dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        ]

        dummy_lm = DummyLM([
            {"answer": "4"},
            {"answer": "blue"}, 
            {"answer": "Paris"},
            {"response": "Improved instruction: Answer questions accurately."}
        ])

        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(simple_metric, max_calls=2)
            
            result = optimizer.compile(student, trainset=trainset)

            assert isinstance(result, dspy.Module)
            assert result._compiled is True
            assert result is not student

    def test_gepa_factory_functions(self):
        """Test GEPA factory functions can be created."""
        # Test that factory functions work
        mute_optimizer = GEPAMute(simple_metric, max_calls=2)
        merge_optimizer = GEPAMerge(simple_metric, max_calls=2)
        
        assert mute_optimizer is not None
        assert merge_optimizer is not None

    def test_optimizer_with_devset(self):
        """Test optimizer with separate dev set."""
        trainset = [
            dspy.Example(question="Train question", answer="train answer").with_inputs("question"),
        ]
        devset = [
            dspy.Example(question="Dev question", answer="dev answer").with_inputs("question"),
        ]

        dummy_lm = DummyLM([
            {"answer": "train answer"},
            {"answer": "dev answer"},
            {"response": "Optimized instruction"}
        ])

        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(simple_metric, max_calls=2)
            
            result = optimizer.compile(student, trainset=trainset, devset=devset)

            assert result._compiled is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])