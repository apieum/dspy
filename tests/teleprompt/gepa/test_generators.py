"""Test ReflectivePromptMutation generator in clean implementation."""

import dspy
from dspy.teleprompt.gepa.generation.reflective_mutation_native import ReflectivePromptMutation
from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Parents, NewBorns
from dspy.teleprompt.gepa.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManager
from unittest.mock import Mock


class TestReflectivePromptMutationGenerator:
    """Test ReflectivePromptMutation generator implementation."""

    def test_generator_interface(self):
        """Test ReflectivePromptMutation implements Generator interface."""
        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)

        # Interface methods
        assert hasattr(generator, 'generate')
        assert hasattr(generator, 'start_compilation')
        assert hasattr(generator, 'finish_compilation')

        # Core generator functionality
        assert hasattr(generator, 'feedback_provider')
        assert hasattr(generator, 'reflection_strategy')

    def test_generator_with_empty_parents(self):
        """Test generator handles empty parents gracefully."""
        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        empty_parents = Parents(iteration=0)
        budget = LMCallsBudget(100)

        result = generator.generate(empty_parents, budget)
        assert isinstance(result, NewBorns)
        assert result.is_empty()

    def test_generator_with_training_data(self):
        """Test generator can be initialized with training data."""
        training_data = [
            dspy.Example(input="test1", answer="answer1").with_inputs("input"),
            dspy.Example(input="test2", answer="answer2").with_inputs("input"),
        ]

        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)

        # Use correct signature with DatasetManager: start_compilation(student, dataset_manager)
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.5)
        generator.start_compilation(dspy.Predict("input -> answer"), dataset_manager)

        # Verify generator has dataset manager and can get feedback minibatch
        assert generator.dataset_manager is dataset_manager
        minibatch = generator.dataset_manager.get_feedback_minibatch(2)
        assert len(minibatch) > 0

    def test_generator_with_parents_but_no_feedback(self):
        """Test generator handles parents without feedback data."""
        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LMCallsBudget(100)

        # No feedback data set
        generator.dev_data = []

        # Create mock parent
        parent_candidate = Candidate(Mock(), generation_number=0)
        parents = Parents(parent_candidate, iteration=0)

        result = generator.generate(parents, budget)
        assert isinstance(result, NewBorns)
        assert result.is_empty()

    def test_generator_budget_consumption(self):
        """Test generator consumes budget appropriately."""
        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LMCallsBudget(100)

        initial_calls = budget.consumed_calls

        # Test empty case consumes budget
        empty_parents = Parents(iteration=0)
        generator.generate(empty_parents, budget)
        assert budget.consumed_calls > initial_calls
