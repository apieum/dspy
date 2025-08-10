"""Test candidate and cohort data structures including ancestry operations."""

import dspy
from unittest.mock import Mock
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import Cohort


class TestCandidate:
    """Test Candidate data structure."""

    def test_candidate_creation(self):
        """Test basic candidate creation."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)

        assert candidate.module == module
        assert candidate.generation_number == 1
        assert candidate.parents == []
        assert candidate.task_scores == {}

    def test_candidate_with_parents(self):
        """Test candidate creation with parent lineage."""
        parent_module = dspy.Predict("input -> output")
        parent = Candidate(parent_module, generation_number=0)

        child_module = dspy.Predict("input -> output")
        child = Candidate(child_module, parents=[parent], generation_number=1)

        assert child.parents == [parent]
        assert child.generation_number == 1

    def test_candidate_task_scoring(self):
        """Test candidate task score management."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)

        # Add task scores (dict format)
        candidate.task_scores = {0: 0.8, 1: 0.6, 2: 0.9}

        assert len(candidate.task_scores) == 3
        assert candidate.task_scores[0] == 0.8
        assert candidate.task_scores[2] == 0.9

    def test_candidate_average_score(self):
        """Test candidate average task score calculation."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)

        # Test with scores (dict format)
        candidate.task_scores = {0: 0.8, 1: 0.6, 2: 0.9, 3: 0.7}
        expected_avg = (0.8 + 0.6 + 0.9 + 0.7) / 4
        assert candidate.average_task_score() == expected_avg

        # Test with empty scores
        candidate.task_scores = {}
        assert candidate.average_task_score() == 0.0

    def test_candidate_task_score_access(self):
        """Test individual task score access."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)

        candidate.task_scores = {0: 0.1, 1: 0.2, 2: 0.3}

        assert candidate.task_score(0) == 0.1
        assert candidate.task_score(1) == 0.2
        assert candidate.task_score(2) == 0.3

        # Test missing key access (returns 0.0 by default)
        assert candidate.task_score(5) == 0.0


class TestCohort:
    """Test Cohort data structure."""

    def test_cohort_creation(self):
        """Test basic cohort creation."""
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)

        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)

        cohort = Cohort(candidate1, candidate2)

        assert cohort.size() == 2
        assert candidate1 in cohort
        assert candidate2 in cohort

    def test_empty_cohort(self):
        """Test empty cohort creation."""
        cohort = Cohort()

        assert cohort.size() == 0
        assert cohort.is_empty()

    def test_cohort_iteration(self):
        """Test cohort candidate access."""
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)

        module2 = dspy.Predict("input -> output")
        candidate2 = Candidate(module2, generation_number=1)

        cohort = Cohort(candidate1, candidate2)
        assert candidate2 in cohort

    def test_cohort_size(self):
        """Test cohort size calculation."""
        module1 = dspy.Predict("input -> output")
        candidate1 = Candidate(module1, generation_number=1)

        cohort = Cohort(candidate1)
        assert cohort.size() == 1

        empty_cohort = Cohort()
        assert empty_cohort.size() == 0


class TestCandidateLineage:
    """Test candidate parent-child relationships."""

    def test_multi_generation_lineage(self):
        """Test multi-generation candidate lineage."""
        # Generation 0 (original)
        gen0_module = dspy.Predict("input -> output")
        gen0_candidate = Candidate(gen0_module, generation_number=0)

        # Generation 1 (child)
        gen1_module = dspy.Predict("input -> output")
        gen1_candidate = Candidate(gen1_module, parents=[gen0_candidate], generation_number=1)

        # Generation 2 (grandchild)
        gen2_module = dspy.Predict("input -> output")
        gen2_candidate = Candidate(gen2_module, parents=[gen1_candidate], generation_number=2)

        # Verify lineage
        assert gen2_candidate.parents == [gen1_candidate]
        assert gen1_candidate.parents == [gen0_candidate]
        assert gen0_candidate.parents == []

        # Verify generation numbers
        assert gen0_candidate.generation_number == 0
        assert gen1_candidate.generation_number == 1
        assert gen2_candidate.generation_number == 2

    def test_multiple_parent_lineage(self):
        """Test candidate with multiple parents (merge)."""
        # Two parent candidates
        parent1_module = dspy.Predict("input -> output")
        parent1 = Candidate(parent1_module, generation_number=1)

        parent2_module = dspy.Predict("input -> output")
        parent2 = Candidate(parent2_module, generation_number=1)

        # Child from merge
        child_module = dspy.Predict("input -> output")
        child = Candidate(child_module, parents=[parent1, parent2], generation_number=2)

        assert len(child.parents) == 2
        assert parent1 in child.parents
        assert parent2 in child.parents
        assert child.generation_number == 2


class TestAncestryEncapsulation:
    """Test the encapsulated ancestry operations."""

    def test_ancestry_operations(self):
        """Test ancestry operations work correctly."""
        # Create test genealogy: ancestor -> parent1, parent2
        ancestor = Candidate(module=Mock(), generation_number=0)
        parent1 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)
        parent2 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)

        # Test the ancestry operations
        common_ancestors = parent1.find_common_ancestors(parent2)

        # Verify the result is correct
        assert ancestor in common_ancestors
        assert len(common_ancestors) == 1

    def test_encapsulation_benefits(self):
        """Test that ancestry implementation details are properly encapsulated."""
        ancestor = Candidate(module=Mock(), generation_number=0)
        parent1 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)
        parent2 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)

        # The Candidate class handles all internal complexity
        common_ancestors = parent1.find_common_ancestors(parent2)

        # The interface provides the needed methods
        assert parent1.is_ancestor_of != None  # Method exists
        assert hasattr(parent1, 'find_common_ancestors')  # Method exists

        # Internal implementation details are hidden
        assert not hasattr(parent1, 'get_ancestors')  # Old method removed from public API

    def test_ancestry_method_behavior(self):
        """Test ancestry method behavior."""
        # Create candidates for testing
        ancestor = Candidate(module=Mock(), generation_number=0)
        parent1 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)
        parent2 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)

        # Test the methods work correctly
        is_ancestor = parent1.is_ancestor_of(parent2)
        common_ancestors = parent1.find_common_ancestors(parent2)

        assert isinstance(is_ancestor, bool)
        assert isinstance(common_ancestors, set)

    def test_maintainability_benefits(self):
        """Test maintainability benefits of encapsulated ancestry logic."""
        # All ancestry logic is encapsulated within the Candidate class
        # Changes to internal implementation don't affect external code

        ancestor = Candidate(module=Mock(), generation_number=0)
        parent = Candidate(module=Mock(), parents=[ancestor], generation_number=1)
        child = Candidate(module=Mock(), parents=[parent], generation_number=2)

        # The public interface is stable
        assert ancestor.is_ancestor_of(child) == True
        assert child.is_ancestor_of(ancestor) == False

        common = parent.find_common_ancestors(child)
        assert ancestor in common

    def test_system_aware_merge_integration(self):
        """Test how SystemAwareMerge uses the ancestry API."""
        ancestor = Candidate(module=Mock(), generation_number=0)
        parent1 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)
        parent2 = Candidate(module=Mock(), parents=[ancestor], generation_number=1)

        # SystemAwareMerge can use clean ancestry operations
        if parent1.is_ancestor_of(parent2) or parent2.is_ancestor_of(parent1):
            assert False, "Should not happen in this test"

        common_ancestors = parent1.find_common_ancestors(parent2)

        assert len(common_ancestors) == 1
        assert ancestor in common_ancestors
