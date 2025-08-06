"""Test paper-compliant generators: ReflectivePromptMutation and SystemAwareMerge."""

import dspy
from dspy.teleprompt.gepa.generation import ReflectivePromptMutation, SystemAwareMerge
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Parents, NewBorns


class TestPaperCompliantGenerators:
    """Test the two final generators from GEPA paper."""

    def test_reflective_prompt_mutation_interface(self):
        """Test ReflectivePromptMutation implements Generator interface."""
        generator = ReflectivePromptMutation()
        
        # Interface methods
        assert hasattr(generator, 'generate')
        assert hasattr(generator, 'start_compilation')
        
        # DSPy Chain-of-Thought components
        assert hasattr(generator, 'performance_analyzer')
        assert hasattr(generator, 'instruction_improver')
        assert hasattr(generator, 'mutate_signature')

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
        generator = ReflectivePromptMutation()
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
        gen1 = ReflectivePromptMutation()
        gen1.start_compilation(dspy.Predict("input -> output"), training_data)
        assert gen1.feedback_data == training_data
        
        # Test SystemAwareMerge
        gen2 = SystemAwareMerge()
        gen2.start_compilation(dspy.Predict("input -> output"), training_data)
        assert gen2.feedback_data == training_data