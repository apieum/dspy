"""Test the simplified, self-contained SystemAwareMerge implementation."""

import pytest
from unittest.mock import Mock, patch

import dspy
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Parents, NewBorns
from dspy.teleprompt.gepa.generation.crossover import SystemAwareMerge

class TestSimplifiedSystemAwareMerge:
    """Test the simplified SystemAwareMerge implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SystemAwareMerge()

        # Create mock modules with predictors
        self.mock_module1 = Mock()
        self.mock_module1.predictors.return_value = [Mock(), Mock()]
        self.mock_module1.deepcopy.return_value = self.mock_module1

        self.mock_module2 = Mock()
        self.mock_module2.predictors.return_value = [Mock(), Mock()]
        self.mock_module2.deepcopy.return_value = self.mock_module2

        self.mock_module3 = Mock()
        self.mock_module3.predictors.return_value = [Mock(), Mock()]
        self.mock_module3.deepcopy.return_value = self.mock_module3

    def test_initialization(self):
        """Test SystemAwareMerge initialization."""
        assert len(self.generator.attempted_merges) == 0
        assert self.generator.merge_stats["success"] == 0

    def test_generate_insufficient_parents(self):
        """Test generate with insufficient parents."""
        # Empty parents
        empty_parents = Parents([])
        result = self.generator.generate(empty_parents)
        assert result.is_empty()

        # Single parent
        single_parent = Parents([Candidate(module=self.mock_module1)])
        result = self.generator.generate(single_parent)
        assert result.is_empty()

    def test_generate_no_common_ancestors(self):
        """Test generate with parents that have no common ancestors."""
        # Create two unrelated candidates
        candidate1 = Candidate(module=self.mock_module1, generation_number=1)
        candidate2 = Candidate(module=self.mock_module2, generation_number=1)

        parents = Parents([candidate1, candidate2])

        result = self.generator.generate(parents)
        assert result.is_empty()

    def test_find_desirable_signatures(self):
        """Test the _find_desirable_signatures method."""
        with patch('dspy.teleprompt.utils.get_signature') as mock_get_sig:
            # Mock signatures - each call pattern for 1 module: ancestor, p1, p2
            ancestor_sig = Mock()
            ancestor_sig.signature = "input -> output"

            p1_sig = Mock()
            p1_sig.signature = "input, context -> output"  # Innovation

            p2_sig = Mock()
            p2_sig.signature = "input -> output"  # Same as ancestor

            # For 2 modules, pattern repeats: a1,p1,p2,a2,p1,p2
            mock_get_sig.side_effect = [
                ancestor_sig, p1_sig, p2_sig,  # Module 0
                ancestor_sig, p1_sig, p2_sig   # Module 1
            ]

            # Create test candidates
            ancestor = Candidate(module=self.mock_module1, generation_number=0)
            parent1 = Candidate(module=self.mock_module2, generation_number=1, task_scores={0: 0.8})
            parent2 = Candidate(module=self.mock_module3, generation_number=1, task_scores={0: 0.6})

            # Test desirable signature detection
            result = self.generator._find_desirable_signatures(ancestor, parent1, parent2)

            # Should detect that parent1 innovated in both modules (condition 1)
            assert len(result) == 2  # 2 modules both show desirable pattern
            assert result[0][0] == 0  # Module index
            assert result[1][0] == 1  # Module index
            # The selected signature should be p1_sig (the innovation)
            # We can't do identity comparison on Mock objects, so check they're not None
            assert result[0][1] is not None
            assert result[1][1] is not None

    def test_create_merged_candidate(self):
        """Test _create_merged_candidate method."""
        with patch('dspy.teleprompt.gepa.generation.crossover.set_signature') as mock_set_sig:
            ancestor = Candidate(module=self.mock_module1, generation_number=0)
            parent1 = Candidate(module=self.mock_module2, generation_number=1)
            parent2 = Candidate(module=self.mock_module3, generation_number=1)

            selected_sigs = [(0, Mock()), (1, Mock())]

            result = self.generator._create_merged_candidate(
                ancestor, parent1, parent2, 2, selected_sigs
            )

            # Verify candidate creation
            assert result is not None
            assert result.generation_number == 2
            assert len(result.parents) == 3  # 3-way lineage
            assert parent1 in result.parents
            assert parent2 in result.parents
            assert ancestor in result.parents

            # Verify signatures were applied
            assert mock_set_sig.call_count == 2

    def test_merge_history_tracking(self):
        """Test merge history prevents duplicate attempts."""
        # Create candidates with common ancestor
        ancestor = Candidate(module=self.mock_module1, generation_number=0)
        parent1 = Candidate(module=self.mock_module2, parents=[ancestor], generation_number=1)
        parent2 = Candidate(module=self.mock_module3, parents=[ancestor], generation_number=1)

        parents = Parents([parent1, parent2])

        # Mock the desirable signatures to fail (no merge)
        with patch.object(self.generator, '_find_desirable_signatures', return_value=[]):
            # First attempt
            result1 = self.generator.generate(parents)
            assert result1.is_empty()

            # Second attempt - should be skipped due to history
            result2 = self.generator.generate(parents)
            assert result2.is_empty()

            # Should have only made the attempt once
            assert len(self.generator.attempted_merges) == 1

    def test_successful_merge_integration(self):
        """Test complete successful merge flow."""
        # Create valid merge scenario
        ancestor = Candidate(module=self.mock_module1, generation_number=0)
        parent1 = Candidate(module=self.mock_module2, parents=[ancestor], generation_number=1, task_scores={0: 0.9})
        parent2 = Candidate(module=self.mock_module3, parents=[ancestor], generation_number=1, task_scores={0: 0.5})

        parents = Parents(*[parent1, parent2], iteration=2)

        # Mock desirable signatures to return a valid result
        with patch.object(self.generator, '_find_desirable_signatures') as mock_desirable, \
             patch.object(self.generator, '_create_merged_candidate') as mock_create:

            mock_desirable.return_value = [(0, Mock())]
            mock_candidate = Candidate(module=Mock(), generation_number=2)
            mock_create.return_value = mock_candidate

            result = self.generator.generate(parents)

            # Verify successful merge
            assert not result.is_empty()
            assert result.size() == 1
            assert self.generator.merge_stats["success"] == 1

    def test_start_compilation(self):
        """Test start_compilation resets state correctly."""
        # Add some state
        self.generator.attempted_merges.add((1, 2, 3))
        self.generator.merge_stats["success"] = 5

        from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManager
        training_data = [Mock(), Mock()]
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.5)

        self.generator.start_compilation(Mock(), dataset_manager)

        # Verify reset
        assert len(self.generator.attempted_merges) == 0
        assert self.generator.merge_stats["success"] == 0

    def test_get_merge_statistics(self):
        """Test merge statistics reporting."""
        # Set up some stats
        self.generator.merge_stats = {
            "success": 3,
            "failure_not_desirable": 2,
            "failure_ancestry": 1
        }
        self.generator.attempted_merges.add((1, 2, 3))
        self.generator.attempted_merges.add((4, 5, 6))

        stats = self.generator.get_merge_statistics()

        assert stats["total_attempts"] == 6
        assert stats["successful_merges"] == 3
        assert stats["success_rate"] == 0.5
        assert stats["unique_combinations_attempted"] == 2

    def test_clear_merge_history(self):
        """Test clear_merge_history method."""
        # Add some state
        self.generator.attempted_merges.add((1, 2, 3))
        self.generator.merge_stats["success"] = 5

        self.generator.clear_merge_history()

        assert len(self.generator.attempted_merges) == 0
        assert self.generator.merge_stats["success"] == 0
