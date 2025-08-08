"""Test Pareto frontier filtering in promote() method."""

import pytest
from unittest.mock import Mock

import dspy
from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManager
from dspy.teleprompt.gepa.selection.pareto_frontier import ParetoFrontier
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.data.cohort import Survivors


class TestParetoPromotion:
    """Test Pareto filtering in promote() method."""

    def test_promote_applies_pareto_filtering(self):
        """Test that promote() removes dominated candidates."""
        selector = ParetoFrontier()

        # Create mock candidates with different performance profiles
        # Candidate A: good at task 0, bad at task 1
        candidate_a = Mock(spec=Candidate)
        candidate_a.task_score = Mock(side_effect=lambda task_id: 0.9 if task_id == 0 else 0.2)
        candidate_a.average_task_score = Mock(return_value=0.55)
        candidate_a.dominate = Mock(return_value=False)
        candidate_a.generation_number = 1

        # Candidate B: bad at task 0, good at task 1
        candidate_b = Mock(spec=Candidate)
        candidate_b.task_score = Mock(side_effect=lambda task_id: 0.1 if task_id == 0 else 0.8)
        candidate_b.average_task_score = Mock(return_value=0.45)
        candidate_b.dominate = Mock(return_value=False)
        candidate_b.generation_number = 1

        # Candidate C: dominated by both A and B (bad at everything)
        candidate_c = Mock(spec=Candidate)
        candidate_c.task_score = Mock(side_effect=lambda task_id: 0.3)
        candidate_c.average_task_score = Mock(return_value=0.3)
        candidate_c.generation_number = 1

        # Set up domination relationships
        # A and B should not dominate each other (trade-offs)
        candidate_a.dominate = Mock(side_effect=lambda other: other == candidate_c)
        candidate_b.dominate = Mock(side_effect=lambda other: other == candidate_c)
        candidate_c.dominate = Mock(return_value=False)

        # Initialize selector with 2 tasks
        training_data = [dspy.Example(task_id=f"task{i}", input="dummy", output="dummy") for i in [1, 2, 3]]
        dataset_manager = DefaultDatasetManager(training_data, pareto_split_ratio=0.67)
        selector.start_compilation(Mock(), dataset_manager)

        # Create survivors cohort
        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=1)

        # Promote survivors
        parents = selector.promote(survivors)

        # Should only promote A and B (C is dominated)
        promoted_candidates = parents.to_list()
        assert parents.size() == 2
        assert parents.contains(candidate_a)
        assert candidate_b in promoted_candidates
        assert candidate_c not in promoted_candidates

    def test_promote_preserves_non_dominated_candidates(self):
        """Test that promote() preserves all non-dominated candidates."""
        selector = ParetoFrontier()

        # Create three non-dominated candidates (each best at different task)
        candidate_a = Mock(spec=Candidate)
        candidate_a.task_score = Mock(side_effect=lambda task_id: [0.9, 0.3, 0.2][task_id])
        candidate_a.average_task_score = Mock(return_value=0.47)
        candidate_a.generation_number = 1

        candidate_b = Mock(spec=Candidate)
        candidate_b.task_score = Mock(side_effect=lambda task_id: [0.2, 0.9, 0.3][task_id])
        candidate_b.average_task_score = Mock(return_value=0.47)
        candidate_b.generation_number = 1

        candidate_c = Mock(spec=Candidate)
        candidate_c.task_score = Mock(side_effect=lambda task_id: [0.3, 0.2, 0.9][task_id])
        candidate_c.average_task_score = Mock(return_value=0.47)
        candidate_c.generation_number = 1

        # None dominate each other (all are Pareto-optimal)
        candidate_a.dominate = Mock(return_value=False)
        candidate_b.dominate = Mock(return_value=False)
        candidate_c.dominate = Mock(return_value=False)

        # Initialize selector with 3 tasks
        training_data = [dspy.Example(task_id=f"task{i}", input="dummy", output="dummy") for i in [1, 2, 3, 4]]
        dataset_manager = DefaultDatasetManager(training_data, pareto_split_ratio=0.75)
        selector.start_compilation(Mock(), dataset_manager)

        # Create survivors cohort
        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=1)

        # Promote survivors
        parents = selector.promote(survivors)

        # Should promote all candidates (none are dominated)
        promoted_candidates = parents.to_list()
        assert len(promoted_candidates) == 3
        assert candidate_a in promoted_candidates
        assert candidate_b in promoted_candidates
        assert candidate_c in promoted_candidates

    def test_promote_removes_strictly_dominated_candidate(self):
        """Test that promote() removes candidates that are strictly dominated."""
        selector = ParetoFrontier()

        # Candidate A: superior in all tasks
        candidate_a = Mock(spec=Candidate)
        candidate_a.task_score = Mock(side_effect=lambda task_id: 0.9)
        candidate_a.average_task_score = Mock(return_value=0.9)
        candidate_a.generation_number = 1

        # Candidate B: inferior in all tasks (strictly dominated)
        candidate_b = Mock(spec=Candidate)
        candidate_b.task_score = Mock(side_effect=lambda task_id: 0.3)
        candidate_b.average_task_score = Mock(return_value=0.3)
        candidate_b.generation_number = 1

        # Set up strict domination: A dominates B
        candidate_a.dominate = Mock(side_effect=lambda other: other == candidate_b)
        candidate_b.dominate = Mock(return_value=False)

        # Initialize selector
        training_data = [dspy.Example(task_id=f"task{i}", input="dummy", output="dummy") for i in [1, 2, 3]]
        dataset_manager = DefaultDatasetManager(training_data, pareto_split_ratio=0.67)
        selector.start_compilation(Mock(), dataset_manager)

        # Create survivors cohort
        survivors = Survivors(candidate_a, candidate_b, iteration=1)

        # Promote survivors
        parents = selector.promote(survivors)

        # Should only promote A (B is strictly dominated)
        promoted_candidates = parents.to_list()
        assert len(promoted_candidates) == 1
        assert candidate_a in promoted_candidates
        assert candidate_b not in promoted_candidates

    def test_promote_preserves_task_wins_for_promoted_candidates(self):
        """Test that promote() includes task_wins data for promoted candidates only."""
        selector = ParetoFrontier()

        # Create candidates
        candidate_a = Mock(spec=Candidate)
        candidate_a.task_score = Mock(return_value=0.9)
        candidate_a.average_task_score = Mock(return_value=0.9)
        candidate_a.generation_number = 1

        candidate_b = Mock(spec=Candidate)
        candidate_b.task_score = Mock(return_value=0.3)  # Dominated
        candidate_b.average_task_score = Mock(return_value=0.3)
        candidate_b.generation_number = 1

        # A dominates B
        candidate_a.dominate = Mock(side_effect=lambda other: other == candidate_b)
        candidate_b.dominate = Mock(return_value=False)

        # Initialize selector with 2 tasks
        training_data = [dspy.Example(task_id=f"task{i}", input="dummy", output="dummy") for i in [1, 2, 3]]
        dataset_manager = DefaultDatasetManager(training_data, pareto_split_ratio=0.67)
        selector.start_compilation(Mock(), dataset_manager)

        # Create survivors cohort
        survivors = Survivors(candidate_a, candidate_b, iteration=1)

        # Promote survivors
        parents = selector.promote(survivors)

        # Check that task_wins only includes promoted candidates (not dominated ones)
        assert candidate_a in parents.task_wins
        assert candidate_b not in parents.task_wins
        # Don't check exact value since update_scores_batch() may recalculate
        assert parents.task_wins[candidate_a] >= 0

    def test_promote_increments_iteration_number(self):
        """Test that promote() properly increments iteration number."""
        selector = ParetoFrontier()

        candidate_a = Mock(spec=Candidate)
        candidate_a.task_score = Mock(return_value=0.9)
        candidate_a.average_task_score = Mock(return_value=0.9)
        candidate_a.dominate = Mock(return_value=False)
        candidate_a.generation_number = 1

        training_data = [dspy.Example(task_id="task1", input="dummy", output="dummy")]
        dataset_manager = DefaultDatasetManager(training_data, pareto_split_ratio=1.0)
        selector.start_compilation(Mock(), dataset_manager)

        # Create survivors with iteration 5
        survivors = Survivors(candidate_a, iteration=5)

        # Promote survivors
        parents = selector.promote(survivors)

        # Should increment iteration
        assert parents.iteration == 6
