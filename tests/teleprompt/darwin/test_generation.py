"""Test Darwin generation components (mutation, reflection, merging)."""

import dspy
from dspy.teleprompt.darwin.generation.mutation import ReflectivePromptMutation
from dspy.teleprompt.darwin.generation.feedback import FeedbackProvider
from dspy.teleprompt.darwin.generation.system_aware_merge import SystemAwareMerge
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy
from unittest.mock import Mock, patch


def simple_metric(example, prediction, trace=None):
    return 0.5


class TestGeneration:
    """Test all generation components."""

    def test_mutation_initialization(self):
        """Test mutation generator initialization."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        feedback_data = [dspy.Example(question="q1", answer="a1").with_inputs("question")]
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=feedback_data
        )

        assert generator.feedback_provider == feedback_provider
        assert generator.feedback_data == feedback_data
        assert generator.module_selection == "round_robin"

    def test_mutation_compilation(self):
        """Test mutation generator compilation setup."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        training_data = [
            dspy.Example(question="q1", answer="a1").with_inputs("question"),
            dspy.Example(question="q2", answer="a2").with_inputs("question"),
        ]
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=training_data
        )

        student = dspy.Predict("question -> answer")
        generator.start_compilation(student, verbose=False)

        assert generator.next_module_idx == 0

    def test_mutation_empty_generation(self):
        """Test mutation handles empty cases."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=[]  # Empty feedback data
        )
        budget = LMCallsBudget(100)

        empty_parents = Parents(iteration=0)
        result = generator.generate(empty_parents, budget)
        assert result.is_empty()

    @patch('dspy.teleprompt.darwin.generation.mutation.ReflectivePromptMutator')
    def test_successful_mutation(self, mutator_mock):
        """Test that a successful mutation returns a new candidate."""
        # Arrange
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        feedback_data = [dspy.Example(question="q1", answer="a1").with_inputs("question")]
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=feedback_data
        )
        student_module = dspy.Predict("question -> answer")
        parent_candidate = Candidate(module=student_module)
        parents = Parents(parent_candidate)
        
        mutated_module = dspy.Predict("question -> mutated_answer")
        mutator_instance = mutator_mock.return_value
        mutator_instance.mutate.return_value = mutated_module

        # Act
        new_borns = generator.generate(parents)

        # Assert
        assert not new_borns.is_empty()
        assert len(new_borns) == 1
        child = list(new_borns)[0]
        assert isinstance(child, Candidate)
        assert child.module is mutated_module
        mutator_instance.mutate.assert_called_once()

    def test_round_robin_module_selection(self):
        """Test that the round robin module selection strategy works correctly."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider)
        
        assert generator._select_target_module(3) == 0
        assert generator._select_target_module(3) == 1
        assert generator._select_target_module(3) == 2
        assert generator._select_target_module(3) == 0

    def test_random_module_selection(self):
        """Test that the random module selection strategy works correctly."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider, module_selection="random")
        
        for _ in range(10):
            module_idx = generator._select_target_module(3)
            assert 0 <= module_idx < 3

    def test_system_aware_merge_initialization(self):
        """Test system aware merge initialization."""
        generator = SystemAwareMerge()
        # Just test that it can be created
        assert generator is not None

    def test_feedback_provider(self):
        """Test feedback provider basic functionality."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        
        example = dspy.Example(question="test", answer="answer")
        prediction = Mock()
        prediction.answer = "answer"
        
        # Test that feedback provider can be created and has the assessor
        assert feedback_provider.assessor == simple_metric
        
        # Test the assessor directly
        score = simple_metric(example, prediction)
        assert score == 0.5  # simple_metric always returns 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
