"""Integration tests for Darwin GEPA optimization."""

import dspy
from dspy.teleprompt.darwin import (
    Darwin, GEPAConfig, GEPAStrategy,
    LMCallsBudget, ParetoFrontier, ReflectivePromptMutation,
    FeedbackProvider, GEPATwoPhasesEval, SystemAwareMerge, ChannelContext, Success,
    GEPAMute, GEPAAdaptive
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

    def test_gepa_lifecycle_completes_within_budget(self):
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        ]
        devset = [
            dspy.Example(question="What color is the sky?", answer="blue").with_inputs("question"),
        ]

        class Observer:
            def __init__(self):
                self.events = []

            def start_compilation(self, student, dataset_manager):
                self.events.append(("start", dataset_manager.num_eval_tasks, dataset_manager.num_dev_examples))

            def finish_compilation(self, result):
                self.events.append(("finish", type(result).__name__))

        observer = Observer()
        dummy_lm = DummyLM([
            {"answer": "4"},
            {"answer": "blue"},
        ])

        with dspy.context(lm=dummy_lm):
            optimizer = Darwin(
                GEPAStrategy,
                GEPAConfig(max_lm_calls=2, observers=(observer,), seed=17),
            )
            compiled = optimizer.compile(SimpleQA(), trainset=trainset, devset=devset)

        assert compiled._compiled is True
        assert observer.events[0] == ("start", 1, 1)
        assert observer.events[-1] == ("finish", "Success")
        assert optimizer.strategy.budget.consumed_calls <= 2
        assert optimizer.get_last_result().history
        assert optimizer.get_last_result().history[0]["evaluated_candidates"] == 1

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
            config = GEPAConfig(
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
        mute_config = GEPAConfig(
            max_lm_calls=2,
            patience=3,
            minibatch_size=3,
            verbose=False
        )

        # Test merge-based configuration
        merge_config = GEPAConfig(
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
            config = GEPAConfig(
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

    def test_public_gepa_optimizers_use_custom_metric(self):
        """Test public GEPA convenience classes compile with old-style metrics."""
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="What color is the sky?", answer="blue").with_inputs("question"),
            dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        ]

        def two_arg_metric(example, prediction):
            expected = example.answer.lower()
            actual = prediction.answer.lower() if hasattr(prediction, "answer") else ""
            return expected == actual

        for optimizer_cls in (GEPAMute, GEPAAdaptive):
            dummy_lm = DummyLM([
                {"answer": "4"},
                {"answer": "blue"},
                {"answer": "Paris"},
                {"answer": "4"},
                {"answer": "blue"},
                {"answer": "Paris"},
            ])

            with dspy.context(lm=dummy_lm):
                optimizer = optimizer_cls(two_arg_metric, max_calls=5, patience=1, minibatch_size=1)
                compiled_module = optimizer.compile(SimpleQA(), trainset=trainset)

                assert compiled_module._compiled is True
                assert isinstance(optimizer.get_last_result(), Success)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
