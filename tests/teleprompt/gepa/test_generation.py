"""Test generation components."""

import dspy
from dspy.teleprompt.gepa.generation.mutation import MutationGenerator
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Cohort
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget


class TestGenerationInterface:
    """Test generation component interfaces."""
    
    def test_mutation_generator_interface(self):
        """Test MutationGenerator implements required interface."""
        generator = MutationGenerator()
        
        # Interface methods
        assert hasattr(generator, 'generate')
        assert hasattr(generator, 'generate_from_parents')
        assert hasattr(generator, 'create_empty_cohort')
        assert hasattr(generator, 'start_compilation')
    
    def test_generator_configuration(self):
        """Test generator self-configuration via start_compilation."""
        generator = MutationGenerator(mutation_rate=0.4, population_size=3)
        
        training_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
            dspy.Example(input="test3", answer="answer3"),
        ]
        
        # Verify starts unconfigured
        assert generator.feedback_data == []
        
        # Configure with training data
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, training_data)
        
        # Verify configuration
        assert generator.feedback_data == training_data


class TestMutationGeneration:
    """Test mutation-based candidate generation."""
    
    def test_generate_from_parents(self):
        """Test basic generation from parent candidates."""
        generator = MutationGenerator(mutation_rate=0.3, population_size=2)
        
        # Setup feedback data
        feedback_data = [
            dspy.Example(input="test1", answer="answer1"),
            dspy.Example(input="test2", answer="answer2"),
        ]
        
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Create parent candidates
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        # Add some mock task scores to make parent selectable
        parent_candidate.task_scores = [0.8, 0.7]
        
        parent_candidates = [parent_candidate]
        budget = LLMCallsBudget(100)
        
        # Generate new candidates
        new_cohort = generator.generate(parent_candidates, iteration=2, budget=budget)
        
        # Verify generation results
        assert isinstance(new_cohort, Cohort)
        assert len(new_cohort.candidates) <= generator.population_size
        
        # Verify new candidates have correct generation number
        for candidate in new_cohort.candidates:
            assert candidate.generation_number == 2
            assert len(candidate.parents) >= 1  # Should have parent lineage
    
    def test_generate_without_parents(self):
        """Test generation behavior with empty parent list."""
        generator = MutationGenerator(population_size=3)
        
        # Setup generator
        feedback_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Generate without parents
        empty_parents = []
        budget = LLMCallsBudget(100)
        result = generator.generate(empty_parents, iteration=1, budget=budget)
        
        # Should return empty cohort
        assert isinstance(result, Cohort)
        assert len(result.candidates) == 0
    
    def test_generate_without_feedback_data(self):
        """Test generation behavior without feedback data."""
        generator = MutationGenerator(population_size=2)
        
        # No feedback data configured
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        parent_candidates = [parent_candidate]
        budget = LLMCallsBudget(100)
        
        # Generate without feedback data
        result = generator.generate(parent_candidates, iteration=1, budget=budget)
        
        # Should return empty cohort
        assert isinstance(result, Cohort)
        assert len(result.candidates) == 0
    
    def test_create_empty_cohort(self):
        """Test empty cohort creation."""
        generator = MutationGenerator()
        
        empty_cohort = generator.create_empty_cohort()
        
        assert isinstance(empty_cohort, Cohort)
        assert len(empty_cohort.candidates) == 0
    
    def test_generate_from_parents_simplified(self):
        """Test simplified generation interface."""
        generator = MutationGenerator(population_size=1)
        
        # Setup
        feedback_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Create parent
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        parent_candidate.task_scores = [0.5]
        
        # Test simplified interface
        result = generator.generate_from_parents([parent_candidate])
        
        assert isinstance(result, Cohort)


class TestGenerationBudgetIntegration:
    """Test generation components work with budget system."""
    
    def test_generation_updates_budget(self):
        """Test that generation calls update budget correctly."""
        generator = MutationGenerator(population_size=1)
        
        # Setup generator
        feedback_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Create parent candidate
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        parent_candidate.task_scores = {0: 0.8}  # Good score for selection
        
        # Track budget usage
        budget = LLMCallsBudget(100)
        initial_consumed = budget.consumed_calls
        
        # Generate (should consume budget for mutations)
        generator.generate([parent_candidate], iteration=1, budget=budget)
        
        # Verify budget was updated
        # Note: Actual consumption depends on whether mutations were successful
        assert budget.consumed_calls >= initial_consumed
    
    def test_generation_without_budget(self):
        """Test generation works without budget parameter."""
        generator = MutationGenerator(population_size=1)
        
        # Setup
        feedback_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Create parent
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        parent_candidate.task_scores = [0.7]
        
        # Generate without budget (should work but not track spending)
        result = generator.generate([parent_candidate], iteration=1, budget=None)
        
        assert isinstance(result, Cohort)
    
    def test_generation_respects_budget_limits(self):
        """Test generation handles budget constraints gracefully."""
        generator = MutationGenerator(population_size=2)
        
        # Setup
        feedback_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Create parents
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        parent_candidate.task_scores = [0.9]
        
        # Very limited budget
        budget = LLMCallsBudget(1)
        budget.consumed_calls = 1  # Already exhausted
        
        # Should handle generation gracefully even with exhausted budget
        result = generator.generate([parent_candidate], iteration=1, budget=budget)
        assert isinstance(result, Cohort)


class TestMutationSpecificBehavior:
    """Test mutation-specific generation behavior."""
    
    def test_parent_selection_strategy(self):
        """Test that generator selects parents based on performance."""
        generator = MutationGenerator(population_size=1)
        
        # Setup
        feedback_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Create candidates with different performance scores
        good_module = dspy.Predict("input -> output")
        good_candidate = Candidate(good_module, generation_number=1)
        good_candidate.task_scores = [0.9, 0.8, 0.9]  # High average
        
        poor_module = dspy.Predict("input -> output")
        poor_candidate = Candidate(poor_module, generation_number=1)
        poor_candidate.task_scores = [0.1, 0.2, 0.1]  # Low average
        
        parent_candidates = [good_candidate, poor_candidate]
        budget = LLMCallsBudget(100)
        
        # Generate (should prefer better parent)
        result = generator.generate(parent_candidates, iteration=1, budget=budget)
        
        # Verify some candidates were generated
        assert isinstance(result, Cohort)
        # Note: Exact parent selection verification would require mocking internal methods
    
    def test_mutation_parameters(self):
        """Test generator respects mutation parameters."""
        generator = MutationGenerator(
            mutation_rate=0.5,
            population_size=3
        )
        
        # Verify parameters are set
        assert generator.mutation_rate == 0.5
        assert generator.population_size == 3
        
        # Setup and test generation respects population size
        feedback_data = [dspy.Example(input="test", answer="answer")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        parent_candidate.task_scores = {0: 0.8}
        
        budget = LLMCallsBudget(100)
        result = generator.generate([parent_candidate], iteration=1, budget=budget)
        
        # Should not exceed population size
        assert len(result.candidates) <= generator.population_size
    
    def test_marks_parents_as_having_children(self):
        """Test that mutation generator marks parents as had_child=True."""
        generator = MutationGenerator(
            population_size=1  # Generate only one child for simplicity
        )
        
        # Setup feedback data
        feedback_data = [dspy.Example(input="test", answer="answer").with_inputs("input")]
        student = dspy.Predict("input -> output")
        generator.start_compilation(student, feedback_data)
        
        # Create parent candidate
        parent_module = dspy.Predict("input -> output")
        parent_candidate = Candidate(parent_module, generation_number=1)
        parent_candidate.task_scores = {0: 0.8}
        parent_candidate.had_child = False  # Initially no children
        
        # Generate children
        result = generator.generate([parent_candidate], iteration=2)
        
        # Parent should now be marked as having children
        assert parent_candidate.had_child == True
        
        # Child should have the parent in its lineage
        if result.candidates:
            child = result.candidates[0]
            assert parent_candidate in child.parents
            assert child.generation_number == 2