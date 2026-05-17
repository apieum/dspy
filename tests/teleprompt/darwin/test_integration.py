"""Integration tests for Darwin GEPA optimization."""

import dspy
from dspy.teleprompt.darwin import (
    Darwin, DarwinConfig, GEPAStrategy,
    LMCallsBudget, ParetoFrontier, ReflectivePromptMutation,
    FeedbackProvider, GEPATwoPhasesEval, SystemAwareMerge, ChannelContext, Success
)
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

            # Create GEPA configuration
            config = DarwinConfig(
                max_lm_calls=2,
                patience=3,
                minibatch_size=3,
                verbose=False
            )

            optimizer = Darwin(GEPAStrategy, config)

            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()

            assert isinstance(result, Success)
            assert isinstance(compiled_module, dspy.Module)
            assert compiled_module._compiled is True
            assert compiled_module is not student

    def test_gepa_configurations(self):
        """Test GEPA configurations can be created."""
        # Test mutation-based configuration
        mute_config = DarwinConfig(
            max_lm_calls=2,
            patience=3,
            minibatch_size=3,
            verbose=False
        )

        # Test merge-based configuration
        merge_config = DarwinConfig(
            mutation=SystemAwareMerge,
            max_lm_calls=2,
            patience=3,
            minibatch_size=3,
            verbose=False
        )

        mute_optimizer = Darwin(GEPAStrategy, mute_config)
        merge_optimizer = Darwin(GEPAStrategy, merge_config)

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

            # Create GEPA configuration
            config = DarwinConfig(
                max_lm_calls=2,
                patience=3,
                minibatch_size=3,
                verbose=False
            )

            optimizer = Darwin(GEPAStrategy, config)

            compiled_module = optimizer.compile(student, trainset=trainset, devset=devset)
            result = optimizer.get_last_result()

            assert isinstance(result, Success)
            assert compiled_module._compiled is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])