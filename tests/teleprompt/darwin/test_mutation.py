"""Test Darwin mutation generators including reflective mutation."""

import dspy
from dspy.teleprompt.darwin.generation.mutation import ReflectivePromptMutation
from dspy.teleprompt.darwin.generation.feedback import FeedbackProvider
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.dataset_manager import DefaultDatasetManager
from unittest.mock import Mock


class TestMutation:
    """Test the simplified ReflectivePromptMutation implementation."""

    def test_basic_initialization(self):
        """Test basic generator initialization."""

        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)

        # Verify initialization
        assert generator.feedback_provider == feedback_provider
        assert generator.minibatch_size == 3  # Default
        assert generator.module_selection == "round_robin"  # Default
        assert generator.dataset_manager is None

    def test_initialization_with_parameters(self):
        """Test generator initialization with custom parameters."""

        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(
            feedback_provider,
            minibatch_size=5,
            module_selection="random"
        )

        # Verify custom parameters
        assert generator.minibatch_size == 5
        assert generator.module_selection == "random"

    def test_start_compilation(self):
        """Test start_compilation properly sets up dataset manager."""

        feedback_provider = FeedbackProvider(metric=lambda ex, pred: 0.5)
        generator = ReflectivePromptMutation(feedback_provider)

        training_data = [
            dspy.Example(question="q1", answer="a1").with_inputs("question"),
            dspy.Example(question="q2", answer="a2").with_inputs("question"),
        ]

        student = dspy.Predict("question -> answer")
        dataset_manager = DefaultDatasetManager(training_data)
        generator.start_compilation(student, dataset_manager)

        # Verify setup
        assert generator.dataset_manager is dataset_manager
        assert generator.next_module_idx == 0

    def test_empty_generation_cases(self):
        """Test generator handles empty cases correctly."""

        feedback_provider = FeedbackProvider(metric=lambda ex, pred: 0.5)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LMCallsBudget(100)

        # Test with empty parents
        empty_parents = Parents(iteration=0)
        result = generator.generate(empty_parents, budget)
        assert result.is_empty()

        # Test with no dataset manager
        generator.dataset_manager = None
        parent_candidate = Candidate(Mock(), generation_number=0)
        parents = Parents(parent_candidate, iteration=0)
        result = generator.generate(parents, budget)
        assert result.is_empty()

    def test_module_selection_strategies(self):
        """Test different module selection strategies."""

        feedback_provider = FeedbackProvider(metric=lambda ex, pred: 0.5)

        # Test round-robin
        generator = ReflectivePromptMutation(feedback_provider, module_selection="round_robin")
        assert generator._select_target_module(3) == 0
        assert generator._select_target_module(3) == 1
        assert generator._select_target_module(3) == 2
        assert generator._select_target_module(3) == 0  # Wraps around

        # Test random (just verify it doesn't crash)
        generator = ReflectivePromptMutation(feedback_provider, module_selection="random")
        result = generator._select_target_module(3)
        assert 0 <= result < 3

    def test_minibatch_sampling(self):
        """Test minibatch sampling through dataset manager."""

        # Setup feedback data through dataset manager
        dev_data = [
            dspy.Example(question=f"q{i}", answer=f"a{i}").with_inputs("question")
            for i in range(5)
        ]
        dataset_manager = DefaultDatasetManager(dev_data)
        dataset_manager = dataset_manager

        # Test sampling
        minibatch = dataset_manager.get_feedback_minibatch(2)
        assert len(minibatch) == 2  # Respects minibatch_size
        assert all(ex in dev_data for ex in minibatch.values())  # Samples from dev_data

        # Test with limited data
        limited_data = dev_data[:1]
        limited_dataset_manager = DefaultDatasetManager(limited_data)
        minibatch = limited_dataset_manager.get_feedback_minibatch(2)
        assert len(minibatch) == 1  # Respects available data

    def test_budget_consumption(self):
        """Test that generator consumes budget appropriately."""

        feedback_provider = FeedbackProvider(metric=lambda ex, pred: 0.5)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LMCallsBudget(100)

        initial_calls = budget.consumed_calls

        # Test empty case consumes budget
        empty_parents = Parents(iteration=0)
        generator.generate(empty_parents, budget)
        assert budget.consumed_calls > initial_calls

    def test_successful_generation_flow(self):
        """Test the complete generation flow when everything works."""

        # This is a simplified test since the full flow involves complex mocking
        feedback_provider = FeedbackProvider(metric=lambda ex, pred: 0.5)
        generator = ReflectivePromptMutation(feedback_provider)

        # Setup
        dev_data = [
            dspy.Example(question="test", answer="answer").with_inputs("question")
        ]
        student = dspy.Predict("question -> answer")
        dataset_manager = DefaultDatasetManager(dev_data)
        generator.start_compilation(student, dataset_manager)

        # Create mock parent with predictors
        parent_module = Mock()
        parent_module.predictors.return_value = [Mock()]  # Has predictors
        parent_candidate = Candidate(parent_module, generation_number=0)
        parents = Parents(parent_candidate, iteration=0)

        budget = LMCallsBudget(100)
        initial_calls = budget.consumed_calls

        # Should attempt generation (may fail due to mocking, but should consume budget)
        result = generator.generate(parents, budget)
        assert budget.consumed_calls > initial_calls  # Budget was consumed


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
