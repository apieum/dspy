"""Test Darwin selector (Pareto frontier selection)."""

import dspy
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import Survivors
from dspy.teleprompt.darwin.selection.pareto import ParetoFrontier
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy


class TestSelector:
    """Test ParetoFrontier selector core functionality."""

    def setup_method(self):
        """Setup fresh selector for each test."""
        self.selector = ParetoFrontier()
        # Need 15 examples to ensure 3 tasks in validation set (20% of 15 = 3)
        training_data = [dspy.Example(task=f"task{i}") for i in range(15)]
        split_strategy = DefaultSplitStrategy(trainset=training_data, verbose=False)
        self.selector.start_compilation(None, split_strategy=split_strategy, verbose=False)

    def create_candidate(self, scores_dict, generation=0):
        """Helper to create candidates with specific scores."""
        candidate = Candidate(
            module=dspy.Predict("input -> output"),
            generation_number=generation
        )
        candidate.task_scores = scores_dict
        return candidate

    def test_basic_pareto_selection(self):
        """Test basic Pareto frontier selection."""
        # Three candidates, each winning different tasks
        candidate_a = self.create_candidate({0: 0.9, 1: 0.5, 2: 0.1})  # Good on task 0
        candidate_b = self.create_candidate({0: 0.2, 1: 0.8, 2: 0.6})  # Good on task 1
        candidate_c = self.create_candidate({0: 0.6, 1: 0.3, 2: 0.95}) # Good on task 2

        # Promote all candidates
        for candidate in [candidate_a, candidate_b, candidate_c]:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Verify task winners
        assert len(self.selector.task_best_candidates[0]) == 1  # A wins task 0
        assert len(self.selector.task_best_candidates[1]) == 1  # B wins task 1
        assert len(self.selector.task_best_candidates[2]) == 1  # C wins task 2

    def test_domination_removal(self):
        """Test that dominated candidates are filtered out."""
        # A dominates B (better on all tasks)
        candidate_a = self.create_candidate({0: 0.9, 1: 0.8, 2: 0.7})  # Superior
        candidate_b = self.create_candidate({0: 0.3, 1: 0.2, 2: 0.1})  # Dominated

        # Verify A dominates B
        assert candidate_a.dominate(candidate_b)

        # Promote both
        survivors = Survivors(candidate_a, candidate_b, iteration=1)
        parents = self.selector.promote(survivors)

        # Only A should be promoted
        assert candidate_a in parents.task_wins
        assert candidate_b not in parents.task_wins

    def test_best_candidate_selection(self):
        """Test best candidate selection by average score."""
        candidate_a = self.create_candidate({0: 0.5, 1: 0.5, 2: 0.5})  # Average: 0.5
        candidate_b = self.create_candidate({0: 0.8, 1: 0.8, 2: 0.8})  # Average: 0.8
        candidate_c = self.create_candidate({0: 0.3, 1: 0.3, 2: 0.3})  # Average: 0.3

        for candidate in [candidate_a, candidate_b, candidate_c]:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        best = self.selector.best_candidate()
        assert best == candidate_b  # Highest average score

    def test_empty_cohort_handling(self):
        """Test handling of empty cohorts."""
        empty_survivors = Survivors(iteration=0)
        parents = self.selector.promote(empty_survivors)
        assert parents.is_empty()

    def test_selection_promotes_better_candidates(self):
        """Test critical bug: selection must properly promote better candidates over worse ones.
        
        This test reproduces a bug where generator expressions in _update_old_winner 
        were not being executed, causing poor candidates to stay in the pool.
        """
        # First, add a poor candidate
        poor_candidate = self.create_candidate({0: 0.1, 1: 0.1, 2: 0.1}, generation=0)
        cohort1 = Survivors(poor_candidate, iteration=0)
        self.selector.promote(cohort1)
        
        # Verify poor candidate is tracked
        assert poor_candidate in self.selector.task_wins
        assert self.selector.task_scores[0] == 0.1  # Poor score for task 0
        
        # Now add a much better candidate for the same task
        good_candidate = self.create_candidate({0: 0.9, 1: 0.2, 2: 0.2}, generation=1)
        cohort2 = Survivors(good_candidate, iteration=1)
        self.selector.promote(cohort2)
        
        # Critical test: good candidate should replace poor one for task 0
        assert good_candidate in self.selector.task_wins
        assert self.selector.task_scores[0] == 0.9  # Updated to good score
        
        # The bug was here: poor_candidate should be removed from task_wins
        # because it was replaced by a better candidate
        if poor_candidate in self.selector.task_wins:
            # Poor candidate might still be in pool if it wins other tasks
            # But it should NOT be winning task 0 anymore
            assert poor_candidate not in self.selector.task_best_candidates[0]
        
        # Best candidate selection should return the good one
        best = self.selector.best_candidate()
        assert best == good_candidate, f"Expected {good_candidate} but got {best}"
        
    def test_task_winner_replacement_with_score_improvement(self):
        """Test that old task winners are properly removed when replaced by better candidates."""
        # Candidate A wins multiple tasks initially (it's the only one)
        candidate_a = self.create_candidate({0: 0.3, 1: 0.8, 2: 0.2}, generation=0)
        cohort1 = Survivors(candidate_a, iteration=0)
        self.selector.promote(cohort1)
        
        assert len(self.selector.task_best_candidates[0]) == 1
        assert candidate_a in self.selector.task_best_candidates[0]
        initial_wins = self.selector.task_wins[candidate_a]  # Wins all 3 tasks initially
        
        # Candidate B comes in with much better score on task 0 only
        candidate_b = self.create_candidate({0: 0.9, 1: 0.1, 2: 0.1}, generation=1)
        cohort2 = Survivors(candidate_b, iteration=1)
        self.selector.promote(cohort2)
        
        # Task 0 should now be won by B only
        assert len(self.selector.task_best_candidates[0]) == 1
        assert candidate_b in self.selector.task_best_candidates[0]
        assert candidate_a not in self.selector.task_best_candidates[0]
        
        # Critical test: A should have lost exactly 1 task win (task 0)
        final_wins_a = self.selector.task_wins.get(candidate_a, 0)
        assert final_wins_a == initial_wins - 1, f"A should lose 1 task win: {initial_wins} -> {final_wins_a}"
        
        # B should win exactly 1 task (task 0)
        assert self.selector.task_wins[candidate_b] == 1  # Wins task 0
        
        # Verify task ownership is correct
        assert candidate_a in self.selector.task_best_candidates[1]  # A still wins task 1
        assert candidate_a in self.selector.task_best_candidates[2]  # A still wins task 2
        assert candidate_b not in self.selector.task_best_candidates[1]  # B doesn't win task 1
        assert candidate_b not in self.selector.task_best_candidates[2]  # B doesn't win task 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])