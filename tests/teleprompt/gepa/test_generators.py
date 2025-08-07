"""Test paper-compliant generators: ReflectivePromptMutation and SystemAwareMerge."""

import dspy
from dspy.teleprompt.gepa.generation import ReflectivePromptMutation, SystemAwareMerge
from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Parents, NewBorns


class TestPaperCompliantGenerators:
    """Test the two final generators from GEPA paper."""

    def test_reflective_prompt_mutation_interface(self):
        """Test ReflectivePromptMutation implements Generator interface."""
        # Define a simple metric for testing
        def simple_metric(example, prediction, trace=None):
            return 1.0 if example.answer == prediction.answer else 0.0
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider)
        
        # Interface methods
        assert hasattr(generator, 'generate')
        assert hasattr(generator, 'start_compilation')
        
        # Core generator functionality
        assert hasattr(generator, 'feedback_provider')
        assert hasattr(generator, 'reflection_strategy')

    def test_system_aware_merge_interface(self):
        """Test SystemAwareMerge implements Generator interface."""
        generator = SystemAwareMerge()
        
        # Interface methods
        assert hasattr(generator, 'generate')
        assert hasattr(generator, 'start_compilation')
        
        # Merge-specific methods
        assert hasattr(generator, 'get_merge_statistics')
        assert hasattr(generator, 'clear_merge_history')

    def test_reflective_prompt_mutation_with_empty_parents(self):
        """Test ReflectivePromptMutation handles empty parents gracefully."""
        def simple_metric(example, prediction, trace=None):
            return 1.0 if example.answer == prediction.answer else 0.0
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider)
        empty_parents = Parents(iteration=0)
        
        result = generator.generate(empty_parents)
        assert isinstance(result, NewBorns)
        assert result.is_empty()

    def test_system_aware_merge_with_insufficient_parents(self):
        """Test SystemAwareMerge handles insufficient parents gracefully.""" 
        generator = SystemAwareMerge()
        
        # Create single parent (need 2 for merge)
        candidate = Candidate(dspy.Predict("input -> output"))
        single_parent = Parents(candidate, iteration=0)
        
        result = generator.generate(single_parent)
        assert isinstance(result, NewBorns)
        assert result.is_empty()

    def test_both_generators_with_training_data(self):
        """Test both generators can be initialized with training data."""
        training_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
        ]
        
        # Test ReflectivePromptMutation
        def simple_metric(example, prediction, trace=None):
            return 1.0 if hasattr(example, 'answer') and hasattr(prediction, 'answer') and \
                   example.answer == prediction.answer else 0.0
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        gen1 = ReflectivePromptMutation(feedback_provider=feedback_provider)
        d_feedback = training_data
        d_pareto = training_data
        gen1.start_compilation(dspy.Predict("input -> output"), d_feedback, d_pareto)
        assert gen1.feedback_data == training_data
        
        # Test SystemAwareMerge
        gen2 = SystemAwareMerge()
        gen2.start_compilation(dspy.Predict("input -> output"), d_feedback, d_pareto)
        assert gen2.feedback_data == training_data