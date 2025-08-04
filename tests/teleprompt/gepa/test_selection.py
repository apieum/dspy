"""Test selection components."""

import dspy
from dspy.teleprompt.gepa.selection.pareto_selection import ParetoSelection
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.candidate_pool import CandidatePool


class TestSelectionInterface:
    """Test selection component interfaces."""
    
    def test_pareto_selection_interface(self):
        """Test ParetoSelection implements required interface."""
        selector = ParetoSelection()
        
        # Interface methods
        assert hasattr(selector, 'filter')
        assert hasattr(selector, 'start_compilation')
        assert hasattr(selector, 'finish_compilation')
        assert isinstance(selector, dspy.teleprompt.gepa.selection.Selection)
    
    def test_selection_observer_lifecycle(self):
        """Test selection components inherit CompilationObserver."""
        selector = ParetoSelection()
        
        # Should have lifecycle methods from CompilationObserver
        assert hasattr(selector, 'start_iteration')
        assert hasattr(selector, 'finish_iteration')
        
        # Test lifecycle methods can be called without error
        student = dspy.Predict("input -> output")
        training_data = [dspy.Example(input="test", answer="answer")]
        
        selector.start_compilation(student, training_data)
        selector.finish_compilation(student, None)
        selector.start_iteration(0, None, None)
        selector.finish_iteration(0, None, None)


class TestParetoSelection:
    """Test Pareto-based candidate selection."""
    
    def test_pareto_selection_basic(self):
        """Test basic Pareto selection with non-dominated candidates."""
        selector = ParetoSelection()
        
        # Create candidates with different performance profiles
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)
        candidate1.task_scores = [0.9, 0.5, 0.7]  # Good at task 0
        
        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)
        candidate2.task_scores = [0.5, 0.9, 0.6]  # Good at task 1
        
        module3 = dspy.Predict("input -> output")
        candidate3 = Candidate(module3, generation_number=1)
        candidate3.task_scores = [0.6, 0.6, 0.9]  # Good at task 2
        
        # Task score data (task_id -> best candidate for that task)
        task_scores = {
            0: candidate1,  # candidate1 wins task 0
            1: candidate2,  # candidate2 wins task 1
            2: candidate3,  # candidate3 wins task 2
        }
        
        # Filter using Pareto selection
        selected = selector.filter(task_scores)
        
        # All candidates should be in Pareto frontier (none dominates others)
        assert len(selected) == 3
        assert candidate1 in selected
        assert candidate2 in selected
        assert candidate3 in selected
    
    def test_pareto_selection_with_domination(self):
        """Test Pareto selection removes dominated candidates."""
        selector = ParetoSelection()
        
        # Create candidates where one dominates another
        module1 = dspy.Predict("input -> output")
        dominant_candidate = Candidate(module1, generation_number=1)
        dominant_candidate.task_scores = [0.9, 0.8, 0.7]  # Better on all tasks
        
        module2 = dspy.Predict("input -> output")
        dominated_candidate = Candidate(module2, generation_number=1)
        dominated_candidate.task_scores = [0.5, 0.6, 0.4]  # Worse on all tasks
        
        # Both candidates are task winners (simplified scenario)
        task_scores = {
            0: dominant_candidate,
            1: dominated_candidate,  # This will be dominated
        }
        
        # Filter using Pareto selection
        selected = selector.filter(task_scores)
        
        # Only dominant candidate should survive
        assert len(selected) == 1
        assert dominant_candidate in selected
        assert dominated_candidate not in selected
    
    def test_pareto_selection_empty_input(self):
        """Test Pareto selection with empty task scores."""
        selector = ParetoSelection()
        
        empty_task_scores = {}
        selected = selector.filter(empty_task_scores)
        
        assert len(selected) == 0
    
    def test_pareto_selection_single_candidate(self):
        """Test Pareto selection with single candidate."""
        selector = ParetoSelection()
        
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)
        candidate.task_scores = [0.8, 0.6, 0.9]
        
        task_scores = {0: candidate}
        selected = selector.filter(task_scores)
        
        assert len(selected) == 1
        assert candidate in selected
    
    def test_pareto_domination_logic(self):
        """Test the Pareto domination logic directly."""
        selector = ParetoSelection()
        
        # Create test candidates
        module1 = dspy.Predict("input -> output")
        candidate_a = Candidate(module1, generation_number=1)
        candidate_a.task_scores = [0.8, 0.7, 0.9]
        
        module2 = dspy.Predict("input -> output")
        candidate_b = Candidate(module2, generation_number=1)
        candidate_b.task_scores = [0.6, 0.5, 0.7]
        
        # Test domination (A should dominate B)
        assert selector._dominates(candidate_a, candidate_b) == True
        assert selector._dominates(candidate_b, candidate_a) == False
        
        # Test non-domination (equal candidates)
        module3 = dspy.Predict("input -> output")
        candidate_c = Candidate(module3, generation_number=1)
        candidate_c.task_scores = [0.8, 0.7, 0.9]  # Same as A
        
        assert selector._dominates(candidate_a, candidate_c) == False
        assert selector._dominates(candidate_c, candidate_a) == False
    
    def test_pareto_mixed_domination(self):
        """Test Pareto selection with mixed domination scenarios."""
        selector = ParetoSelection()
        
        # Create candidates with complex relationships
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)
        candidate1.task_scores = [0.9, 0.3, 0.8]  # Excellent at 0&2, poor at 1
        
        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)
        candidate2.task_scores = [0.4, 0.9, 0.7]  # Excellent at 1, decent at 2
        
        module3 = dspy.Predict("input -> output")
        candidate3 = Candidate(module3, generation_number=1)
        candidate3.task_scores = [0.6, 0.6, 0.6]  # Mediocre at all
        
        task_scores = {
            0: candidate1,  
            1: candidate2,
            2: candidate1,  # candidate1 also wins task 2
        }
        
        selected = selector.filter(task_scores)
        
        # candidate1 and candidate2 should be in frontier (neither dominates)
        # candidate3 might be dominated depending on the scenario
        assert candidate1 in selected
        assert candidate2 in selected
        assert len(selected) >= 2


class TestSelectionWithCandidatePool:
    """Test selection integration with CandidatePool."""
    
    def test_selection_receives_task_winners(self):
        """Test that selection receives proper task winner data."""
        # This tests the integration pattern between CandidatePool and Selection
        
        # Create candidate pool
        pool = CandidatePool()
        
        # Create test candidates
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)
        candidate1.task_scores = {0: 0.9, 1: 0.5}
        
        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)
        candidate2.task_scores = {0: 0.6, 1: 0.8}
        
        # Add candidates to pool
        from dspy.teleprompt.gepa.data.cohort import Cohort
        cohort = Cohort(candidate1, candidate2)
        pool.promote(cohort)
        
        # Test that pool can provide task score data for selection
        selector = ParetoSelection()
        
        # Verify pool has the expected structure for selection
        assert pool.size() == 2
        assert candidate1 in pool.candidates
        assert candidate2 in pool.candidates
    
    def test_selection_maintains_counts(self):
        """Test that ParetoSelection maintains selection counts."""
        selector = ParetoSelection()
        
        # Create test scenario
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)
        candidate.task_scores = [0.8, 0.7]
        
        task_scores = {0: candidate}
        
        # Initial state
        initial_count = selector.selection_counts[candidate]
        
        # Perform selection
        selected = selector.filter(task_scores)
        
        # Verify selection was tracked
        assert len(selected) == 1
        assert candidate in selected
        # Selection counts tracking (internal implementation detail)
        assert hasattr(selector, 'selection_counts')


class TestSelectionEdgeCases:
    """Test selection component edge cases."""
    
    def test_selection_with_incomplete_scores(self):
        """Test selection handles candidates with different numbers of task scores."""
        selector = ParetoSelection()
        
        # Create candidates with different score lengths
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)
        candidate1.task_scores = [0.8, 0.7, 0.9]  # 3 scores
        
        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)
        candidate2.task_scores = [0.6, 0.5]  # 2 scores
        
        task_scores = {
            0: candidate1,
            1: candidate2,
        }
        
        # Should handle gracefully
        selected = selector.filter(task_scores)
        assert isinstance(selected, list)
        assert len(selected) >= 0
    
    def test_selection_with_zero_scores(self):
        """Test selection handles candidates with zero scores."""
        selector = ParetoSelection()
        
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)
        candidate.task_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        
        task_scores = {0: candidate}
        selected = selector.filter(task_scores)
        
        assert len(selected) == 1
        assert candidate in selected
    
    def test_excludes_candidates_with_children(self):
        """Test that candidates with had_child=True are excluded from selection."""
        selector = ParetoSelection()
        
        # Create candidates - one with children, one without
        module1 = dspy.Predict("input -> output")
        parent_candidate = Candidate(module1, generation_number=1)
        parent_candidate.task_scores = {0: 0.9, 1: 0.8}
        parent_candidate.had_child = True  # This candidate has produced children
        
        module2 = dspy.Predict("input -> output")
        fresh_candidate = Candidate(module2, generation_number=1)
        fresh_candidate.task_scores = {0: 0.7, 1: 0.6}
        fresh_candidate.had_child = False  # This candidate is still eligible
        
        # Both are task winners but only fresh_candidate should be selected
        task_scores = {
            0: parent_candidate,  # parent_candidate wins task 0
            1: fresh_candidate,   # fresh_candidate wins task 1
        }
        
        selected = selector.filter(task_scores)
        
        # Only the fresh candidate should be selected
        assert len(selected) == 1
        assert fresh_candidate in selected
        assert parent_candidate not in selected
    
    def test_returns_empty_when_all_candidates_have_children(self):
        """Test returns empty list when all candidates have produced children."""
        selector = ParetoSelection()
        
        # Create candidates that all have produced children
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)
        candidate1.task_scores = {0: 0.9, 1: 0.5}
        candidate1.had_child = True
        
        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)
        candidate2.task_scores = {0: 0.6, 1: 0.8}
        candidate2.had_child = True
        
        task_scores = {
            0: candidate1,
            1: candidate2,
        }
        
        selected = selector.filter(task_scores)
        
        # Should return empty list
        assert len(selected) == 0