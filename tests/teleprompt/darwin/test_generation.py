"""Test Darwin generation components (mutation, reflection, merging)."""

import dspy
from dspy.teleprompt.darwin.generation.mutation import ReflectivePromptMutation
from dspy.teleprompt.darwin.generation.feedback import FeedbackProvider
from dspy.teleprompt.darwin.generation.system_aware_merge import SystemAwareMerge
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy
from unittest.mock import Mock


def simple_metric(example, prediction, trace=None):
    return 0.5


class TestGeneration:
    """Test all generation components."""

    def test_mutation_initialization(self):
        """Test mutation generator initialization."""
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)

        assert generator.feedback_provider == feedback_provider
        assert generator.minibatch_size == 3
        assert generator.module_selection == "round_robin"

    def test_mutation_compilation(self):
        """Test mutation generator compilation setup."""
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)

        training_data = [
            dspy.Example(question="q1", answer="a1").with_inputs("question"),
            dspy.Example(question="q2", answer="a2").with_inputs("question"),
        ]

        student = dspy.Predict("question -> answer")
        split_strategy = DefaultSplitStrategy(trainset=training_data, verbose=False)
        generator.start_compilation(student, split_strategy=split_strategy, verbose=False)

        assert generator.next_module_idx == 0

    def test_mutation_empty_generation(self):
        """Test mutation handles empty cases."""
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LMCallsBudget(100)

        empty_parents = Parents(iteration=0)
        result = generator.generate(empty_parents, budget)
        assert result.is_empty()

    def test_system_aware_merge_initialization(self):
        """Test system aware merge initialization."""
        generator = SystemAwareMerge()
        # Just test that it can be created
        assert generator is not None

    def test_feedback_provider(self):
        """Test feedback provider basic functionality."""
        feedback_provider = FeedbackProvider(metric=simple_metric)
        
        example = dspy.Example(question="test", answer="answer")
        prediction = Mock()
        prediction.answer = "answer"
        
        # Test that feedback provider can be created and has the metric
        assert feedback_provider.metric == simple_metric
        
        # Test the metric directly
        score = simple_metric(example, prediction)
        assert score == 0.5  # simple_metric always returns 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])