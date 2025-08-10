"""Comprehensive tests for Darwin selector components.

Tests all scenarios and edge cases for selection strategies including Pareto frontier
selection with task winner logic, domination handling, and integration scenarios.
"""

import dspy
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import Survivors, Parents
from dspy.teleprompt.darwin.selection.pareto import ParetoFrontier
from dspy.teleprompt.darwin.dataset_manager import DefaultDatasetManager


class TestSelector:
    """Comprehensive tests for ParetoFrontier with all scenarios."""

    def setup_method(self):
        """Setup fresh selector for each test."""
        self.selector = ParetoFrontier()
        # Create mock training data with 3 tasks
        d_feedback = [dspy.Example(task=f"task{i}") for i in range(3)]
        d_pareto = [dspy.Example(task=f"task{i}") for i in range(3)]

        # Combine training data and create DatasetManager
        combined_training_data = d_feedback + d_pareto
        dataset_manager = DefaultDatasetManager(combined_training_data, split_ratio=0.5)

        self.selector.start_compilation(None, dataset_manager)

    def create_candidate(self, scores_dict, generation=0, parents=None):
        """Helper to create candidates with specific scores."""
        candidate = Candidate(
            module=dspy.Predict("input -> output"),
            generation_number=generation,
            parents=parents or []
        )
        candidate.task_scores = scores_dict
        return candidate

    def test_basic_algorithm_2_compliance(self):
        """Test basic Algorithm 2 compliance with diverse performance profiles."""
        # Create candidates matching the paper examples
        candidate_a = self.create_candidate({0: 0.9, 1: 0.5, 2: 0.1})  # Good on task 0
        candidate_b = self.create_candidate({0: 0.2, 1: 0.8, 2: 0.6})  # Good on task 1
        candidate_c = self.create_candidate({0: 0.6, 1: 0.3, 2: 0.95}) # Good on task 2
        candidate_d = self.create_candidate({0: 0.9, 1: 0.2, 2: 0.1})  # Tied with A on task 0, dominated
        candidate_e = self.create_candidate({0: 0.7, 1: 0.4, 2: 0.05}) # Dominated by A

        candidates = [candidate_a, candidate_b, candidate_c, candidate_d, candidate_e]

        # Step 1: Promote all candidates
        for candidate in candidates:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Verify task winners are correctly identified
        assert len(self.selector.task_best_candidates[0]) == 1  # Only A (dominates D)
        assert len(self.selector.task_best_candidates[1]) == 1  # Only B
        assert len(self.selector.task_best_candidates[2]) == 1  # Only C

        assert candidate_a in self.selector.task_best_candidates[0]
        assert candidate_b in self.selector.task_best_candidates[1]
        assert candidate_c in self.selector.task_best_candidates[2]

        # Step 2: Test Algorithm 2 Pareto frontier selection
        survivors = Survivors(*candidates, iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        frontier_candidates = set()
        for candidate in pareto_frontier:
            frontier_candidates.add(candidate)
        expected_frontier = {candidate_a, candidate_b, candidate_c}

        assert frontier_candidates == expected_frontier
        assert len(pareto_frontier) == 3

    def test_tied_winners_handling(self):
        """Test proper handling of tied winners on specific tasks."""
        # Create candidates with exact ties
        candidate_a = self.create_candidate({0: 0.9, 1: 0.5, 2: 0.1})
        candidate_b = self.create_candidate({0: 0.9, 1: 0.3, 2: 0.8})  # Tied with A on task 0
        candidate_c = self.create_candidate({0: 0.7, 1: 0.8, 2: 0.2})  # Best on task 1

        # Promote all candidates
        cohort = Survivors(candidate_a, candidate_b, candidate_c, iteration=0)
        self.selector.promote(cohort)

        # Task 0 should have both A and B (tied at 0.9)
        task_0_winners = self.selector.task_best_candidates[0]
        assert len(task_0_winners) == 2
        assert candidate_a in task_0_winners
        assert candidate_b in task_0_winners

        # Task 1 should only have C (0.8 > 0.5, 0.3)
        task_1_winners = self.selector.task_best_candidates[1]
        assert len(task_1_winners) == 1
        assert candidate_c in task_1_winners

        # All should be in Pareto frontier (no domination)
        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents
        assert len(pareto_frontier) == 3

    def test_domination_exclusion(self):
        """Test that dominated candidates are properly excluded."""
        # A dominates B on all tasks
        candidate_a = self.create_candidate({0: 0.9, 1: 0.8, 2: 0.7})  # Superior
        candidate_b = self.create_candidate({0: 0.8, 1: 0.7, 2: 0.6})  # Dominated by A
        candidate_c = self.create_candidate({0: 0.5, 1: 0.9, 2: 0.4})  # Good on task 1

        cohort = Survivors(candidate_a, candidate_b, candidate_c, iteration=0)
        self.selector.promote(cohort)

        # Verify domination relationship
        assert candidate_a.dominate(candidate_b)

        # A should be the only task 0 winner (dominates B)
        assert self.selector.task_best_candidates[0] == [candidate_a]
        # C should be the only task 1 winner
        assert self.selector.task_best_candidates[1] == [candidate_c]
        # A should be the only task 2 winner (dominates B)
        assert self.selector.task_best_candidates[2] == [candidate_a]

        # Pareto frontier should only have A and C (B is dominated)
        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        frontier_set = set()
        for candidate in pareto_frontier:
            frontier_set.add(candidate)
        assert frontier_set == {candidate_a, candidate_c}
        assert candidate_b not in frontier_set

    def test_dominated_candidate_wins_single_task(self):
        """Test when a Pareto-dominated candidate is the best on one specific task.

        Scenario:
        - Candidate A: Task 0=1.0, Task 1=0.9, Task 2=0.9 (dominates B)
        - Candidate B: Task 0=1.0, Task 1=0.1, Task 2=0.1 (tied on task 0 but dominated overall)

        When tied on task score, domination should matter.
        """
        # Create candidates - A dominates B (at least as good on all, better on tasks 1&2)
        cand_a = self.create_candidate({0: 1.0, 1: 0.9, 2: 0.9})
        cand_b = self.create_candidate({0: 1.0, 1: 0.1, 2: 0.1})

        # Verify A dominates B
        assert cand_a.dominate(cand_b), "A should dominate B"
        assert not cand_b.dominate(cand_a), "B should not dominate A"

        # Update scores in selector
        self.selector.update_score(0, cand_a)
        self.selector.update_score(0, cand_b)

        # Check task 0 winner - both tie on score (1.0) but A dominates B
        task_0_winners = self.selector.task_best_candidates[0]
        assert cand_a in task_0_winners, "A should be in task 0 winners (scores 1.0 and dominates)"
        assert cand_b not in task_0_winners, "B should be replaced by A when tied on task but dominated overall"
        assert len(task_0_winners) == 1, "Should have exactly 1 task winner (A replaces B due to domination)"

    def test_better_task_score_wins_regardless_of_domination(self):
        """Test that better task-specific performance wins regardless of domination status.

        Scenario:
        - Candidate A: Task 0=0.8, Task 1=0.9, Task 2=0.9
        - Candidate B: Task 0=1.0, Task 1=0.7, Task 2=0.7

        Neither dominates the other, but B clearly wins task 0.
        Key principle: Task winners are determined by task-specific scores, not overall domination.
        """
        # Create candidates - neither dominates, but B is better on task 0
        cand_a = self.create_candidate({0: 0.8, 1: 0.9, 2: 0.9})
        cand_b = self.create_candidate({0: 1.0, 1: 0.7, 2: 0.7})

        # Verify neither dominates the other
        assert not cand_a.dominate(cand_b), "A should not dominate B"
        assert not cand_b.dominate(cand_a), "B should not dominate A"

        # B should win task 0 due to better performance
        assert cand_b.task_score(0) > cand_a.task_score(0), "B should score better on task 0 (1.0 > 0.8)"

        # Update scores
        self.selector.update_score(0, cand_a)
        self.selector.update_score(0, cand_b)

        # B should win task 0 with higher score
        task_0_winners = self.selector.task_best_candidates[0]
        assert cand_b in task_0_winners, "B should win task 0 with score 1.0 > 0.8"
        assert cand_a not in task_0_winners, "A should not win task 0 with inferior score 0.8 < 1.0"

    def test_equal_task_scores_without_domination(self):
        """Test when candidates have exactly equal scores on a task with no domination.

        Scenario:
        - Candidate A: Task 0=1.0, Task 1=0.8, Task 2=0.9
        - Candidate B: Task 0=1.0, Task 1=0.9, Task 2=0.7

        Both score 1.0 on task 0, neither dominates the other.
        Both should be kept as task winners.
        """
        cand_a = self.create_candidate({0: 1.0, 1: 0.8, 2: 0.9})
        cand_b = self.create_candidate({0: 1.0, 1: 0.9, 2: 0.7})

        # Verify neither dominates the other
        assert not cand_a.dominate(cand_b), "A should not dominate B"
        assert not cand_b.dominate(cand_a), "B should not dominate A"

        # Both have equal scores on task 0
        assert cand_a.task_score(0) == cand_b.task_score(0), "Both should have equal scores on task 0"

        # Update scores
        self.selector.update_score(0, cand_a)
        self.selector.update_score(0, cand_b)

        # Both should be kept as task winners (no domination to break the tie)
        task_0_winners = self.selector.task_best_candidates[0]
        assert cand_a in task_0_winners, "A should be kept as task 0 winner"
        assert cand_b in task_0_winners, "B should be kept as task 0 winner"
        assert len(task_0_winners) == 2, "Both candidates should be task winners"

    def test_equal_task_scores_with_domination(self):
        """Test when candidates have equal scores on a task but one dominates the other."""
        cand_a = self.create_candidate({0: 1.0, 1: 0.9, 2: 0.9})
        cand_b = self.create_candidate({0: 1.0, 1: 0.1, 2: 0.1})

        # Verify A dominates B
        assert cand_a.dominate(cand_b), "A should dominate B"

        # Both have equal scores on task 0
        assert cand_a.task_score(0) == cand_b.task_score(0), "Both should have equal scores on task 0 (1.0)"

        # Update scores
        self.selector.update_score(0, cand_a)
        self.selector.update_score(0, cand_b)

        # Only the dominating candidate should remain
        task_0_winners = self.selector.task_best_candidates[0]
        assert cand_a in task_0_winners, "A (dominating) should be task winner"
        assert cand_b not in task_0_winners, "B (dominated) should be replaced"
        assert len(task_0_winners) == 1, "Only dominating candidate should remain"

    def test_task_winner_replacement_with_domination(self):
        """Test that task winners are properly replaced when better candidates arrive.

        Scenario progression:
        1. Candidate A becomes task 0 winner with score 0.8
        2. Candidate B arrives with score 0.9 on task 0 (replaces A)
        3. Candidate C arrives with score 0.9 on task 0 but dominates B

        Question: Should C replace B as task winner when they tie on the task?
        """
        # Step 1: A becomes initial winner
        cand_a = self.create_candidate({0: 0.8, 1: 0.5, 2: 0.5})
        self.selector.update_score(0, cand_a)
        assert len(self.selector.task_best_candidates[0]) == 1
        assert cand_a in self.selector.task_best_candidates[0]

        # Step 2: B replaces A with better score
        cand_b = self.create_candidate({0: 0.9, 1: 0.3, 2: 0.3})
        self.selector.update_score(0, cand_b)
        assert cand_b in self.selector.task_best_candidates[0], "B should be task winner with score 0.9"
        assert cand_a not in self.selector.task_best_candidates[0], "A should be replaced (0.8 < 0.9)"

        # Step 3: C arrives with same task score but dominates B
        cand_c = self.create_candidate({0: 0.9, 1: 0.5, 2: 0.5})
        assert cand_c.dominate(cand_b), "C should dominate B (better on tasks 1 and 2)"

        self.selector.update_score(0, cand_c)

        # C should replace B due to domination when tied on task performance
        task_0_winners = self.selector.task_best_candidates[0]
        assert cand_c in task_0_winners, "C should be in task 0 winners (score 0.9)"

        # Check if domination causes replacement
        if len(task_0_winners) == 1:
            assert task_0_winners[0] == cand_c, "When tied and dominating, C should be sole winner"
        else:
            # Both might be kept as tied winners despite domination
            assert cand_b in task_0_winners, "B might remain as tied winner despite domination"

    def test_ancestry_based_replacement(self):
        """Test that children replace their parents in task winner lists."""
        # Parent candidate
        parent = self.create_candidate({0: 0.8, 1: 0.6, 2: 0.4}, generation=0)

        # Child with identical performance but newer generation
        child = self.create_candidate({0: 0.8, 1: 0.6, 2: 0.4}, generation=1, parents=[parent])

        # Promote parent first
        parent_cohort = Survivors(parent, iteration=0)
        self.selector.promote(parent_cohort)

        # Verify parent is in all task winner lists
        for task_id in range(3):
            assert parent in self.selector.task_best_candidates[task_id]

        # Promote child - should replace parent due to ancestry
        child_cohort = Survivors(child, iteration=0)
        self.selector.promote(child_cohort)

        # Child should replace parent in all task winner lists
        for task_id in range(3):
            assert child in self.selector.task_best_candidates[task_id]
            assert parent not in self.selector.task_best_candidates[task_id]

    def test_merge_generation_ancestry(self):
        """Test ancestry handling for merge generation (multiple parents)."""
        # Two parents with complementary strengths
        parent1 = self.create_candidate({0: 0.9, 1: 0.3, 2: 0.2}, generation=0)
        parent2 = self.create_candidate({0: 0.3, 1: 0.9, 2: 0.2}, generation=0)

        # Merge child inheriting from both parents
        merge_child = self.create_candidate(
            {0: 0.7, 1: 0.7, 2: 0.8},
            generation=1,
            parents=[parent1, parent2]
        )

        # Promote all candidates
        for candidate in [parent1, parent2, merge_child]:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Task 0: parent1 should win (0.9 > 0.7 > 0.3)
        assert self.selector.task_best_candidates[0] == [parent1]

        # Task 1: parent2 should win (0.9 > 0.7 > 0.3)
        assert self.selector.task_best_candidates[1] == [parent2]

        # Task 2: merge_child should win (0.8 > 0.2)
        assert self.selector.task_best_candidates[2] == [merge_child]


    def test_empty_cohort_handling(self):
        """Test behavior with empty cohorts and no candidates."""
        # Test promoting empty cohort
        empty_cohort = Survivors(iteration=0)
        result = self.selector.promote(empty_cohort)

        assert len(result) == 0
        assert self.selector.size() == 0
        # task_scores should still have the initialized tasks with 0.0 scores
        assert all(score == 0.0 for score in self.selector.task_scores.values())

        # Test filtering with no promoted candidates
        survivors = Survivors(iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        assert len(pareto_frontier) == 0

    def test_single_candidate_scenarios(self):
        """Test behavior with single candidates."""
        single_candidate = self.create_candidate({0: 0.5, 1: 0.7, 2: 0.3})

        # Promote single candidate
        cohort = Survivors(single_candidate, iteration=0)
        self.selector.promote(cohort)

        # Should be winner on all tasks
        for task_id in range(3):
            assert self.selector.task_best_candidates[task_id] == [single_candidate]

        # Should be in Pareto frontier
        survivors = Survivors(single_candidate, iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        assert len(pareto_frontier) == 1
        assert single_candidate in pareto_frontier

    def test_best_candidate_selection(self):
        """Test best_candidate() method returns correct overall best."""
        # Create candidates with different average scores
        candidate_a = self.create_candidate({0: 0.9, 1: 0.1, 2: 0.1})  # avg = 0.37
        candidate_b = self.create_candidate({0: 0.6, 1: 0.6, 2: 0.6})  # avg = 0.60 (best)
        candidate_c = self.create_candidate({0: 0.1, 1: 0.9, 2: 0.1})  # avg = 0.37

        for candidate in [candidate_a, candidate_b, candidate_c]:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        best = self.selector.best_candidate()
        assert best == candidate_b  # Highest average score

    def test_stochastic_sampling(self):
        """Test stochastic sampling functionality."""
        # Override with 2-task setup
        training_data = [dspy.Example(task=f"task{i}") for i in range(2)]
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.5)
        self.selector.start_compilation(None, dataset_manager)

        # Create multiple good candidates
        candidates = [
            self.create_candidate({0: 0.9, 1: 0.1}),
            self.create_candidate({0: 0.1, 1: 0.9}),
            self.create_candidate({0: 0.8, 1: 0.8})
        ]

        cohort = Survivors(*candidates, iteration=0)
        parents = self.selector.promote(cohort)

        # Test stochastic sampling
        sampled = parents.sample_stochastic(2)

        assert len(sampled) <= 2
        assert len(sampled) >= 1  # Should return at least 1
        # Verify all sampled candidates are from original set
        for candidate in sampled:
            assert candidate in candidates

    def test_complex_multi_task_scenario(self):
        """Test complex scenario with 5 tasks and 8 candidates."""
        # Override with 5-task setup - create enough data to ensure 5 pareto tasks
        training_data = [dspy.Example(task=f"task{i}") for i in range(10)]  # 10 examples
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.6)  # 6 pareto tasks, but we'll only use first 5
        self.selector.start_compilation(None, dataset_manager)

        # Create diverse candidates with different specializations
        candidates = [
            self.create_candidate({0: 0.95, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}),  # Task 0 specialist
            self.create_candidate({0: 0.1, 1: 0.95, 2: 0.1, 3: 0.1, 4: 0.1}),  # Task 1 specialist
            self.create_candidate({0: 0.1, 1: 0.1, 2: 0.95, 3: 0.1, 4: 0.1}),  # Task 2 specialist
            self.create_candidate({0: 0.1, 1: 0.1, 2: 0.1, 3: 0.95, 4: 0.1}),  # Task 3 specialist
            self.create_candidate({0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.95}),  # Task 4 specialist
            self.create_candidate({0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6, 4: 0.6}),   # Generalist
            self.create_candidate({0: 0.8, 1: 0.2, 2: 0.8, 3: 0.2, 4: 0.8}),   # Alternating pattern
            self.create_candidate({0: 0.7, 1: 0.7, 2: 0.1, 3: 0.1, 4: 0.1})    # Tasks 0,1 specialist
        ]

        # Promote all candidates
        for candidate in candidates:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Verify each task has correct winner
        for task_id in range(5):
            winners = self.selector.task_best_candidates[task_id]
            assert len(winners) == 1
            assert winners[0] == candidates[task_id]  # Task specialists should win

        # Test Pareto frontier - should include all specialists (no domination)
        survivors = Survivors(*candidates, iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        # Specialists shouldn't dominate each other
        assert len(pareto_frontier) >= 5  # At least the 5 specialists

    def test_performance_with_large_candidate_set(self):
        """Test performance and correctness with larger candidate sets."""
        # Override with 4-task setup
        training_data = [dspy.Example(task=f"task{i}") for i in range(4)]
        dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.5)
        self.selector.start_compilation(None, dataset_manager)

        # Create 20 candidates with random-ish but controlled performance
        candidates = []
        for i in range(20):
            # Create performance profile where each candidate is best at task i%4
            scores = {j: 0.3 + (0.6 if j == i % 4 else 0.0) + (i * 0.01) for j in range(4)}
            candidates.append(self.create_candidate(scores, generation=i//4))

        # Promote all candidates
        for candidate in candidates:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Verify system can handle large sets - filter candidates from generation 4 (last generation)
        last_gen_candidates = [c for c in candidates if c.generation_number == 4]
        survivors = Survivors(*last_gen_candidates, iteration=5)  # iteration 5, so gen 4 candidates are eligible
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        assert len(pareto_frontier) >= 1
        assert len(pareto_frontier) <= len(last_gen_candidates)

        # Verify all frontier candidates are non-dominated
        frontier_list = list(pareto_frontier)
        for i, candidate_a in enumerate(frontier_list):
            for j, candidate_b in enumerate(frontier_list):
                if i != j:
                    # No candidate in frontier should dominate another
                    assert not candidate_a.dominate(candidate_b)

    def test_edge_case_zero_scores(self):
        """Test handling of zero scores."""
        # Candidates with zero scores
        candidate_a = self.create_candidate({0: 0.0, 1: 0.5, 2: 1.0})
        candidate_b = self.create_candidate({0: 0.1, 1: 0.0, 2: 0.5})
        candidate_c = self.create_candidate({0: 0.0, 1: 0.0, 2: 0.0})

        for candidate in [candidate_a, candidate_b, candidate_c]:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Should handle zero scores correctly
        assert candidate_b in self.selector.task_best_candidates[0]  # 0.1 > 0.0
        assert candidate_a in self.selector.task_best_candidates[1]  # 0.5 > 0.0
        assert candidate_a in self.selector.task_best_candidates[2]  # 1.0 is best

        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        # A and B should be in frontier, C is dominated by both
        assert len(pareto_frontier) == 2
        frontier_set = set()
        for candidate in pareto_frontier:
            frontier_set.add(candidate)
        assert candidate_c not in frontier_set

    def test_incremental_promotion(self):
        """Test incremental promotion behavior over multiple iterations."""
        # Start with initial candidates
        initial_candidates = [
            self.create_candidate({0: 0.5, 1: 0.8, 2: 0.3}, generation=0),
            self.create_candidate({0: 0.7, 1: 0.4, 2: 0.9}, generation=0)
        ]

        # Promote initial generation
        for candidate in initial_candidates:
            cohort = Survivors(candidate, iteration=0)
            promoted = self.selector.promote(cohort)
            assert promoted.iteration == 1  # Iteration should increment

        # Add improved candidates later
        improved_candidates = [
            self.create_candidate({0: 0.9, 1: 0.9, 2: 0.5}, generation=1),
            self.create_candidate({0: 0.6, 1: 0.6, 2: 0.95}, generation=1)
        ]

        for candidate in improved_candidates:
            cohort = Survivors(candidate, iteration=1)
            self.selector.promote(cohort)

        # Verify improved candidates become task winners
        assert improved_candidates[0] in self.selector.task_best_candidates[0]  # 0.9 best
        assert improved_candidates[0] in self.selector.task_best_candidates[1]  # 0.9 best
        assert improved_candidates[1] in self.selector.task_best_candidates[2]  # 0.95 best

    def test_algorithm_2_step_by_step_trace(self):
        """Detailed step-by-step trace of Algorithm 2 execution."""
        # Setup exactly as described in paper
        candidate_a = self.create_candidate({0: 0.9, 1: 0.5, 2: 0.1}, generation=0)
        candidate_b = self.create_candidate({0: 0.2, 1: 0.8, 2: 0.6}, generation=0)
        candidate_c = self.create_candidate({0: 0.6, 1: 0.3, 2: 0.95}, generation=0)

        # Algorithm 2 Step 1: For each τ ∈ D, find s*[i] = max_k S_P[k][i]
        for candidate in [candidate_a, candidate_b, candidate_c]:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Verify max scores per task
        assert max(c.task_score(0) for c in [candidate_a, candidate_b, candidate_c]) == 0.9  # A wins task 0
        assert max(c.task_score(1) for c in [candidate_a, candidate_b, candidate_c]) == 0.8  # B wins task 1
        assert max(c.task_score(2) for c in [candidate_a, candidate_b, candidate_c]) == 0.95 # C wins task 2

        # Algorithm 2 Step 2: P*[i] = {P[k] : S_P[k][i] = s*[i]}
        assert self.selector.task_best_candidates[0] == [candidate_a]
        assert self.selector.task_best_candidates[1] == [candidate_b]
        assert self.selector.task_best_candidates[2] == [candidate_c]

        # Algorithm 2 Step 3: C = unique candidates in ∪_i P*[i]
        all_task_winners = set()
        for task_winners in self.selector.task_best_candidates.values():
            all_task_winners.update(task_winners)

        assert all_task_winners == {candidate_a, candidate_b, candidate_c}

        # Algorithm 2 Step 4: Remove dominated candidates from C
        survivors = Survivors(*[candidate_a, candidate_b, candidate_c], iteration=1)
        pareto_parents = self.selector.promote(survivors)
        pareto_frontier = pareto_parents

        # Final result: all three should be in frontier (no domination)
        frontier_set = set()
        for candidate in pareto_frontier:
            frontier_set.add(candidate)
        assert frontier_set == {candidate_a, candidate_b, candidate_c}
        assert len(pareto_frontier) == 3

    def test_promote_increments_iteration_number(self):
        """Test that promote() properly increments iteration number."""
        single_candidate = self.create_candidate({0: 0.8, 1: 0.6, 2: 0.4})
        
        # Create survivors with iteration 5
        survivors = Survivors(single_candidate, iteration=5)
        
        # Promote survivors
        parents = self.selector.promote(survivors)
        
        # Should increment iteration
        assert parents.iteration == 6

    def test_promote_preserves_task_wins_for_promoted_candidates(self):
        """Test that promote() includes task_wins data for promoted candidates only."""
        # Create candidates where one dominates the other
        candidate_a = self.create_candidate({0: 0.9, 1: 0.8, 2: 0.7})  # Superior
        candidate_b = self.create_candidate({0: 0.3, 1: 0.2, 2: 0.1})  # Dominated by A
        
        # Verify A dominates B
        assert candidate_a.dominate(candidate_b)
        
        # Create survivors cohort
        survivors = Survivors(candidate_a, candidate_b, iteration=1)
        
        # Promote survivors
        parents = self.selector.promote(survivors)
        
        # Check that task_wins only includes promoted candidates (not dominated ones)
        assert candidate_a in parents.task_wins
        assert candidate_b not in parents.task_wins
        # Verify task_wins has valid data
        assert parents.task_wins[candidate_a] >= 0
