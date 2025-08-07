"""Test simplified ReflectivePromptMutation generator."""

import dspy
from dspy.teleprompt.gepa.generation.reflective_mutation_native import ReflectivePromptMutation
from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Parents, NewBorns
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget
from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManager
from unittest.mock import Mock


class TestSimplifiedReflectivePromptMutation:
    """Test simplified ReflectivePromptMutation implementation."""
    
    def test_generator_basic_functionality(self):
        """Test basic generator functionality."""
        def simple_metric(example, prediction, trace=None):
            return 0.5
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        
        # Test interface
        assert hasattr(generator, 'generate')
        assert hasattr(generator, 'start_compilation')
        assert hasattr(generator, 'feedback_provider')
        assert hasattr(generator, 'reflection_strategy')
    
    def test_generator_with_empty_parents(self):
        """Test generator handles empty parents gracefully."""
        def simple_metric(example, prediction, trace=None):
            return 0.5
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LLMCallsBudget(100)
        
        empty_parents = Parents(iteration=0)
        result = generator.generate(empty_parents, budget)
        
        assert isinstance(result, NewBorns)
        assert result.is_empty()
    
    def test_generator_start_compilation(self):
        """Test generator start_compilation with correct signature."""
        def simple_metric(example, prediction, trace=None):
            return 0.5
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        
        # Set up training data
        training_data = [
            dspy.Example(input="test1", answer="answer1").with_inputs("input"),
            dspy.Example(input="test2", answer="answer2").with_inputs("input"),
        ]
        
        # Use correct signature with DatasetManager: start_compilation(student, dataset_manager)
        dataset_manager = DefaultDatasetManager(training_data, pareto_split_ratio=0.5)
        generator.start_compilation(dspy.Predict("input -> answer"), dataset_manager)
        
        # Verify setup
        assert generator.dataset_manager is dataset_manager
        minibatch = generator.dataset_manager.get_feedback_minibatch(1)
        assert len(minibatch) > 0
    
    def test_generator_with_parents(self):
        """Test generator behavior with parent candidates."""
        def simple_metric(example, prediction, trace=None):
            return 0.5
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LLMCallsBudget(100)
        
        # Set up feedback data
        training_data = [
            dspy.Example(input="test1", answer="answer1").with_inputs("input")
        ]
        generator.start_compilation(dspy.Predict("input -> answer"), training_data)
        
        # Create parent candidate with predictors
        parent_module = Mock()
        parent_module.predictors.return_value = [Mock()]  # Has predictors
        candidate = Candidate(parent_module, generation_number=0)
        parents = Parents(candidate, iteration=0)
        
        initial_calls = budget.consumed_calls
        result = generator.generate(parents, budget)
        
        # Should consume budget (generation may succeed or fail, but budget should be used)
        assert budget.consumed_calls > initial_calls
        assert isinstance(result, NewBorns)
    
    def test_generator_without_feedback_data(self):
        """Test generator behavior without feedback data."""
        def simple_metric(example, prediction, trace=None):
            return 0.5
        
        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        budget = LLMCallsBudget(100)
        
        # No feedback data set
        generator.feedback_data = []
        
        # Create mock parent
        parent_candidate = Candidate(Mock(), generation_number=0)
        parents = Parents(parent_candidate, iteration=0)
        
        initial_calls = budget.consumed_calls
        result = generator.generate(parents, budget)
        
        # Should consume budget and return empty
        assert budget.consumed_calls > initial_calls
        assert isinstance(result, NewBorns)
        assert result.is_empty()


if __name__ == "__main__":
    test = TestSimplifiedReflectivePromptMutation()
    test.test_generator_basic_functionality()
    test.test_generator_with_empty_parents()
    test.test_generator_start_compilation()
    test.test_generator_with_parents()
    test.test_generator_without_feedback_data()
    print("All simplified reflection tests passed!")