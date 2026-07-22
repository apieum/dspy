"""Comprehensive tests for ParetoFrontier selector - the core GEPA Algorithm 2 implementation.

This test suite validates the official GEPA Algorithm 2 implementation with:
- UUID-based task identification 
- Pareto dominance filtering
- Task winner accumulation and replacement
- Edge cases and error conditions
- Observer pattern integration
"""

import pytest
import asyncio
import random
from unittest.mock import Mock, AsyncMock
import dspy
from dspy.teleprompt.darwin.selection.pareto import ParetoFrontier
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import Survivors, Parents
from dspy.teleprompt.darwin.evaluation.metrics import Metric
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin import DarwinConfig


class TestParetoFrontierCore:
    """Test core ParetoFrontier functionality."""

    def setup_method(self):
        """Setup fresh selector for each test."""
        self.selector = ParetoFrontier()
        self.selector.start_compilation(None, verbose=False)

    def create_candidate_with_scores(self, scores_dict, generation=0, candidate_id=None):
        """Helper to create candidates with UUID-based scores."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=generation)
        
        # Use custom ID if provided (for deterministic testing)
        if candidate_id is not None:
            # Override the candidate's id for testing
            candidate._test_id = candidate_id
        
        # Convert scores_dict to Metric objects with proper UUID system
        candidate.scores = [
            Metric(value, id=f'uuid-{task_id}', trace={'dspy_uuid': f'uuid-{task_id}'})
            for task_id, value in scores_dict.items()
        ]
        return candidate

    def test_initialization(self):
        """Test ParetoFrontier initialization."""
        selector = ParetoFrontier()
        
        assert selector.example_best_scores == {}
        assert selector.example_best_candidates == {}
        assert len(selector.task_wins) == 0
        assert hasattr(selector, 'publish')  # Channel capability
        assert selector.elitist_pruning == False  # Default mode is official GEPA

    def test_start_compilation_resets_previous_state(self):
        candidate = self.create_candidate_with_scores({"task": 1.0})
        self.selector.update_scores_batch(Survivors(candidate, iteration=0))
        assert self.selector.size() == 1

        self.selector.configure(DarwinConfig(preserve_diversity=True, archive_capacity=4))
        self.selector.diversity_archive.add(candidate)
        self.selector.start_compilation(None, verbose=False)

        assert self.selector.size() == 0
        assert self.selector.example_best_scores == {}
        assert self.selector.example_best_candidates == {}
        assert len(self.selector.diversity_archive) == 0

    def test_stochastic_sampling_can_select_without_replacement(self):
        candidates = [self.create_candidate_with_scores({"task": 1.0}) for _ in range(3)]
        cohort = Parents(*candidates, task_wins={candidate: index + 1 for index, candidate in enumerate(candidates)})

        selected = cohort.sample_stochastic(2, rng=random.Random(7), replace=False)

        assert len(selected.candidates) == 2

    def test_single_candidate_promotion(self):
        """Test promoting a single candidate."""
        candidate = self.create_candidate_with_scores({0: 0.8, 1: 0.6, 2: 0.9})
        survivors = Survivors(candidate, iteration=0)
        
        parents = self.selector.promote(survivors)
        
        assert parents.size() == 1
        assert candidate in parents.candidates
        assert self.selector.task_wins[candidate] == 3  # Wins all examples
        assert len(self.selector.example_best_candidates) == 3

    def test_multiple_candidates_different_strengths(self):
        """Test promotion with candidates having different strengths."""
        # Each candidate excels at different examples
        candidate_a = self.create_candidate_with_scores({0: 0.9, 1: 0.3, 2: 0.2})  # Best at example 0
        candidate_b = self.create_candidate_with_scores({0: 0.2, 1: 0.8, 2: 0.3})  # Best at example 1  
        candidate_c = self.create_candidate_with_scores({0: 0.3, 1: 0.2, 2: 0.95}) # Best at example 2
        
        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=0)
        parents = self.selector.promote(survivors)
        
        # All should be promoted (non-dominated)
        assert parents.size() == 3
        assert {candidate_a, candidate_b, candidate_c} == set(parents.candidates)
        
        # Each wins one example
        assert self.selector.task_wins[candidate_a] == 1
        assert self.selector.task_wins[candidate_b] == 1  
        assert self.selector.task_wins[candidate_c] == 1
        
        # Verify example ownership
        assert candidate_a in self.selector.example_best_candidates['uuid-0']
        assert candidate_b in self.selector.example_best_candidates['uuid-1']
        assert candidate_c in self.selector.example_best_candidates['uuid-2']

    def test_pareto_domination_filtering(self):
        """Test that dominated candidates are filtered out."""
        # Candidate A dominates B (better on all examples)
        candidate_a = self.create_candidate_with_scores({0: 0.9, 1: 0.8, 2: 0.7})  # Superior
        candidate_b = self.create_candidate_with_scores({0: 0.3, 1: 0.2, 2: 0.1})  # Dominated
        
        # Verify A dominates B
        assert candidate_a.dominate(candidate_b)
        
        survivors = Survivors(candidate_a, candidate_b, iteration=0)
        parents = self.selector.promote(survivors)
        
        # Only A should be promoted
        assert parents.size() == 1
        assert candidate_a in parents.candidates
        assert candidate_b not in parents.candidates
        
        # A wins all examples
        assert self.selector.task_wins[candidate_a] == 3
        assert candidate_b not in self.selector.task_wins

    def test_task_winner_replacement(self):
        """Test that better candidates replace worse ones as task winners."""
        # First candidate wins everything initially
        poor_candidate = self.create_candidate_with_scores({0: 0.3, 1: 0.3, 2: 0.3})
        survivors1 = Survivors(poor_candidate, iteration=0)
        self.selector.promote(survivors1)
        
        assert self.selector.task_wins[poor_candidate] == 3
        
        # Better candidate for example 0 only
        good_candidate = self.create_candidate_with_scores({0: 0.9, 1: 0.1, 2: 0.1})
        survivors2 = Survivors(good_candidate, iteration=1)
        parents = self.selector.promote(survivors2)
        
        # Good candidate should take over example 0
        assert good_candidate in self.selector.example_best_candidates['uuid-0']
        assert poor_candidate not in self.selector.example_best_candidates['uuid-0']
        
        # Poor candidate keeps examples 1 and 2
        assert poor_candidate in self.selector.example_best_candidates['uuid-1']
        assert poor_candidate in self.selector.example_best_candidates['uuid-2']
        
        # Task win counts updated correctly
        assert self.selector.task_wins[poor_candidate] == 2  # Lost one example
        assert self.selector.task_wins[good_candidate] == 1  # Won one example

    def test_tied_scores_with_domination(self):
        """Test handling of tied scores with domination analysis."""
        # First candidate
        candidate_a = self.create_candidate_with_scores({0: 0.5, 1: 0.8, 2: 0.3})
        survivors1 = Survivors(candidate_a, iteration=0)
        self.selector.promote(survivors1)
        
        # Second candidate with same score on example 0 but dominates overall
        candidate_b = self.create_candidate_with_scores({0: 0.5, 1: 0.9, 2: 0.4})  # Better on examples 1,2
        survivors2 = Survivors(candidate_b, iteration=1)
        self.selector.promote(survivors2)
        
        # B should dominate A and take over example 1 and 2
        assert candidate_b.dominate(candidate_a)
        
        # In official GEPA mode, B should be the sole winner for examples 1 and 2
        assert candidate_b in self.selector.example_best_candidates['uuid-1']
        assert candidate_b in self.selector.example_best_candidates['uuid-2']
        
        # For example 0 (tied), in official mode both candidates coexist
        assert candidate_b in self.selector.example_best_candidates['uuid-0']
        # A might still be present if it tied on example 0
        
        # B should have more task wins overall
        assert self.selector.task_wins[candidate_b] >= 2

    def test_ancestry_based_replacement(self):
        """Test that evolved children and parents can coexist in tied scenarios (Official GEPA mode)."""
        # Parent candidate
        parent = self.create_candidate_with_scores({0: 0.7, 1: 0.7}, generation=0)
        survivors1 = Survivors(parent, iteration=0)
        self.selector.promote(survivors1)
        
        # Child with same scores but newer generation
        child = self.create_candidate_with_scores({0: 0.7, 1: 0.7}, generation=1)
        child.parents = [parent]  # Set ancestry
        survivors2 = Survivors(child, iteration=1)
        self.selector.promote(survivors2)
        
        # In official GEPA mode, both child and parent can coexist for tied scores
        assert child in self.selector.task_wins
        # Parent might still be present if scores are truly tied
        
        # Both should be able to appear in example winners for tied cases
        assert child in self.selector.example_best_candidates['uuid-0'] or parent in self.selector.example_best_candidates['uuid-0']
        assert child in self.selector.example_best_candidates['uuid-1'] or parent in self.selector.example_best_candidates['uuid-1']
        
        # Total task wins should be distributed appropriately
        total_wins = sum(self.selector.task_wins.values())
        assert total_wins >= 2  # At least as many wins as examples

    def test_best_candidate_selection(self):
        """Test best candidate selection by average score."""
        candidate_low = self.create_candidate_with_scores({0: 0.3, 1: 0.3})   # avg: 0.3
        candidate_high = self.create_candidate_with_scores({0: 0.9, 1: 0.9})  # avg: 0.9
        candidate_mid = self.create_candidate_with_scores({0: 0.6, 1: 0.6})   # avg: 0.6
        
        survivors = Survivors(candidate_low, candidate_high, candidate_mid, iteration=0)
        self.selector.promote(survivors)
        
        best = self.selector.best_candidate()
        assert best == candidate_high  # Highest average score

    def test_best_candidate_no_candidates_error(self):
        """Test error when no candidates are available for selection."""
        with pytest.raises(RuntimeError, match="No candidates found in selector"):
            self.selector.best_candidate()

    def test_empty_survivors_promotion(self):
        """Test promoting empty survivors cohort."""
        empty_survivors = Survivors(iteration=0)
        parents = self.selector.promote(empty_survivors)
        
        assert parents.is_empty()
        assert len(self.selector.task_wins) == 0

    # Tests merged from test_selector.py
    def test_basic_pareto_selection_merged(self):
        """Test basic Pareto frontier selection with UUID-based system."""
        # Three candidates, each winning different examples
        candidate_a = self.create_candidate_with_scores({0: 0.9, 1: 0.5, 2: 0.1})  # Good on example 0
        candidate_b = self.create_candidate_with_scores({0: 0.2, 1: 0.8, 2: 0.6})  # Good on example 1
        candidate_c = self.create_candidate_with_scores({0: 0.6, 1: 0.3, 2: 0.95}) # Good on example 2

        # Promote all candidates
        for candidate in [candidate_a, candidate_b, candidate_c]:
            cohort = Survivors(candidate, iteration=0)
            self.selector.promote(cohort)

        # Verify example winners using UUID-based system
        assert len(self.selector.example_best_candidates['uuid-0']) == 1  # A wins example 0
        assert len(self.selector.example_best_candidates['uuid-1']) == 1  # B wins example 1
        assert len(self.selector.example_best_candidates['uuid-2']) == 1  # C wins example 2

    def test_domination_removal_merged(self):
        """Test that dominated candidates are filtered out."""
        # A dominates B (better on all tasks)
        candidate_a = self.create_candidate_with_scores({0: 0.9, 1: 0.8, 2: 0.7})  # Superior
        candidate_b = self.create_candidate_with_scores({0: 0.3, 1: 0.2, 2: 0.1})  # Dominated

        # Verify A dominates B
        assert candidate_a.dominate(candidate_b)

        # Promote both
        survivors = Survivors(candidate_a, candidate_b, iteration=1)
        parents = self.selector.promote(survivors)

        # Only A should be promoted (B is dominated)
        assert candidate_a in parents.candidates
        assert candidate_b not in parents.candidates  # B is dominated and filtered out
        
        # A should be in task_wins
        assert candidate_a in self.selector.task_wins
        assert candidate_b not in self.selector.task_wins

    def test_selection_promotes_better_candidates_merged(self):
        """Test critical feature: selection must properly promote better candidates over worse ones.
        
        This test verifies the UUID-based task winner replacement system.
        """
        # First, add a poor candidate
        poor_candidate = self.create_candidate_with_scores({0: 0.1, 1: 0.1, 2: 0.1}, generation=0)
        cohort1 = Survivors(poor_candidate, iteration=0)
        self.selector.promote(cohort1)
        
        # Verify poor candidate wins examples initially (it's the only one)
        assert poor_candidate in self.selector.task_wins
        assert self.selector.task_wins[poor_candidate] == 3  # Wins all 3 examples initially
        
        # Now add a much better candidate for example 0, but worse for examples 1 and 2
        good_candidate = self.create_candidate_with_scores({0: 0.9, 1: 0.05, 2: 0.05}, generation=1)
        cohort2 = Survivors(good_candidate, iteration=1)
        self.selector.promote(cohort2)
        
        # Critical test: good candidate should replace poor one for example 0
        assert good_candidate in self.selector.task_wins
        
        # Good candidate should win example 0, poor candidate should win examples 1 and 2
        assert good_candidate in self.selector.example_best_candidates['uuid-0']
        assert poor_candidate not in self.selector.example_best_candidates['uuid-0']
        assert poor_candidate in self.selector.example_best_candidates['uuid-1']  # Still wins example 1
        assert poor_candidate in self.selector.example_best_candidates['uuid-2']  # Still wins example 2
        
        # Best candidate selection should return the good one (higher average)
        best = self.selector.best_candidate()
        assert best == good_candidate, f"Expected {good_candidate} but got {best}"
        
    def test_task_winner_replacement_with_score_improvement_merged(self):
        """Test that old task winners are properly removed when replaced by better candidates."""
        # Candidate A wins multiple examples initially (it's the only one)
        candidate_a = self.create_candidate_with_scores({0: 0.3, 1: 0.8, 2: 0.2}, generation=0)
        cohort1 = Survivors(candidate_a, iteration=0)
        self.selector.promote(cohort1)
        
        assert len(self.selector.example_best_candidates['uuid-0']) == 1
        assert candidate_a in self.selector.example_best_candidates['uuid-0']
        initial_wins = self.selector.task_wins[candidate_a]  # Wins all 3 examples initially
        
        # Candidate B comes in with much better score on example 0 only
        candidate_b = self.create_candidate_with_scores({0: 0.9, 1: 0.1, 2: 0.1}, generation=1)
        cohort2 = Survivors(candidate_b, iteration=1)
        self.selector.promote(cohort2)
        
        # Example 0 should now be won by B only
        assert len(self.selector.example_best_candidates['uuid-0']) == 1
        assert candidate_b in self.selector.example_best_candidates['uuid-0']
        assert candidate_a not in self.selector.example_best_candidates['uuid-0']
        
        # Critical test: A should have lost exactly 1 example win (example 0)
        final_wins_a = self.selector.task_wins.get(candidate_a, 0)
        assert final_wins_a == initial_wins - 1, f"A should lose 1 example win: {initial_wins} -> {final_wins_a}"
        
        # B should win exactly 1 example (example 0)
        assert self.selector.task_wins[candidate_b] == 1  # Wins example 0
        
        # Verify example ownership is correct
        assert candidate_a in self.selector.example_best_candidates['uuid-1']  # A still wins example 1
        assert candidate_a in self.selector.example_best_candidates['uuid-2']  # A still wins example 2
        assert candidate_b not in self.selector.example_best_candidates['uuid-1']  # B doesn't win example 1
        assert candidate_b not in self.selector.example_best_candidates['uuid-2']  # B doesn't win example 2

class TestParetoFrontierGEPAModes:
    """Test Official GEPA mode vs Elitist pruning mode behavior differences."""

    def setup_method(self):
        self.selector = ParetoFrontier()
        self.selector.start_compilation(None, verbose=False)

    def create_candidate_with_scores(self, scores_dict, generation=0):
        """Helper to create candidates with UUID-based scores."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=generation)
        candidate.scores = [
            Metric(value, id=f'uuid-{task_id}', trace={'dspy_uuid': f'uuid-{task_id}'})
            for task_id, value in scores_dict.items()
        ]
        return candidate

    def test_official_gepa_mode_default(self):
        """Test that Official GEPA mode is the default."""
        assert self.selector.elitist_pruning == False

    def test_official_gepa_mode_tied_scores(self):
        """Test Official GEPA mode: tied scores result in multiple winners."""
        # Ensure we're in official GEPA mode
        self.selector.elitist_pruning = False
        
        # First candidate
        candidate_a = self.create_candidate_with_scores({0: 0.7, 1: 0.5})
        survivors1 = Survivors(candidate_a, iteration=0)
        self.selector.promote(survivors1)
        
        # Second candidate with same score on example 0
        candidate_b = self.create_candidate_with_scores({0: 0.7, 1: 0.3})
        survivors2 = Survivors(candidate_b, iteration=1)
        self.selector.promote(survivors2)
        
        # In official GEPA mode, both should coexist as winners for example 0
        example_0_candidates = self.selector.example_best_candidates['uuid-0']
        assert candidate_a in example_0_candidates
        assert candidate_b in example_0_candidates
        assert len(example_0_candidates) == 2
        
        # Both should have task wins for example 0
        assert self.selector.task_wins[candidate_a] >= 1  # Wins example 0 + 1
        assert self.selector.task_wins[candidate_b] >= 1  # Wins example 0

    def test_elitist_pruning_mode_tied_scores(self):
        """Test Elitist pruning mode: ties are resolved by generation or other criteria."""
        # Enable elitist pruning mode
        self.selector.elitist_pruning = True
        
        # First candidate (older generation)
        candidate_a = self.create_candidate_with_scores({0: 0.7, 1: 0.5}, generation=0)
        survivors1 = Survivors(candidate_a, iteration=0)
        self.selector.promote(survivors1)
        
        # Second candidate with same score but newer generation
        candidate_b = self.create_candidate_with_scores({0: 0.7, 1: 0.3}, generation=1)
        survivors2 = Survivors(candidate_b, iteration=1)
        self.selector.promote(survivors2)
        
        # In elitist mode, behavior may differ for tie handling
        # This test verifies that elitist pruning affects tie handling
        example_0_candidates = self.selector.example_best_candidates.get('uuid-0', [])
        assert len(example_0_candidates) >= 1  # At least one winner
        
        # Both candidates should still exist in task wins if they have different strengths
        total_task_wins = sum(self.selector.task_wins.values())
        assert total_task_wins >= 2  # At least 2 example wins distributed

    def test_mode_switching_behavior(self):
        """Test that mode can be switched and affects subsequent operations."""
        # Start in official GEPA mode
        assert self.selector.elitist_pruning == False
        
        # Add some candidates
        candidate_a = self.create_candidate_with_scores({0: 0.5, 1: 0.8})
        survivors1 = Survivors(candidate_a, iteration=0)
        self.selector.promote(survivors1)
        
        # Switch to elitist mode
        self.selector.elitist_pruning = True
        assert self.selector.elitist_pruning == True
        
        # Add more candidates - should use elitist behavior
        candidate_b = self.create_candidate_with_scores({0: 0.5, 1: 0.6})
        survivors2 = Survivors(candidate_b, iteration=1)
        self.selector.promote(survivors2)
        
        # Verify both modes worked (mode switching doesn't break state)
        assert len(self.selector.task_wins) >= 1
        assert len(self.selector.example_best_candidates) >= 1

    def test_official_gepa_mode_multiple_ties(self):
        """Test Official GEPA mode with multiple tied candidates."""
        self.selector.elitist_pruning = False
        
        # Three candidates all tied on example 0
        candidate_a = self.create_candidate_with_scores({0: 0.6, 1: 0.5, 2: 0.3})
        candidate_b = self.create_candidate_with_scores({0: 0.6, 1: 0.4, 2: 0.7})
        candidate_c = self.create_candidate_with_scores({0: 0.6, 1: 0.8, 2: 0.2})
        
        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=0)
        parents = self.selector.promote(survivors)
        
        # All should be winners for example 0 in official GEPA mode
        example_0_candidates = self.selector.example_best_candidates['uuid-0']
        assert candidate_a in example_0_candidates
        assert candidate_b in example_0_candidates
        assert candidate_c in example_0_candidates
        assert len(example_0_candidates) == 3
        
        # Each should win their respective best examples
        assert candidate_c in self.selector.example_best_candidates['uuid-1']  # Best at example 1
        assert candidate_b in self.selector.example_best_candidates['uuid-2']  # Best at example 2


class TestParetoFrontierEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        self.selector = ParetoFrontier()
        self.selector.start_compilation(None, verbose=False)

    def create_candidate_with_scores(self, scores_dict, generation=0):
        """Helper to create candidates with UUID-based scores."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=generation)
        candidate.scores = [
            Metric(value, id=f'uuid-{task_id}', trace={'dspy_uuid': f'uuid-{task_id}'})
            for task_id, value in scores_dict.items()
        ]
        return candidate

    def test_candidates_with_no_scores(self):
        """Test handling candidates with no scores."""
        candidate = Candidate(dspy.Predict("input -> output"))
        # No scores assigned
        survivors = Survivors(candidate, iteration=0)
        
        parents = self.selector.promote(survivors)
        
        # Should handle gracefully
        assert parents.size() == 0  # No scores = no promotion
        assert candidate not in self.selector.task_wins

    def test_candidates_use_metric_ids_without_uuid_traces(self):
        """Test handling candidates whose score UUIDs are stored on Metric.id."""
        candidate = Candidate(dspy.Predict("input -> output"))
        candidate.scores = [
            Metric(0.8, id='uuid-empty-trace', trace={}),
            Metric(0.6, id='uuid-none-trace', trace={'dspy_uuid': None}),
            Metric(0.7, id='uuid-no-trace'),
        ]
        
        survivors = Survivors(candidate, iteration=0)
        parents = self.selector.promote(survivors)
        
        assert isinstance(parents, Parents)
        assert self.selector.task_wins[candidate] == 3
        assert 'uuid-empty-trace' in self.selector.example_best_candidates
        assert 'uuid-none-trace' in self.selector.example_best_candidates
        assert 'uuid-no-trace' in self.selector.example_best_candidates

    def test_candidates_with_zero_scores_are_promoted(self):
        """Test first zero-valued scores still initialize example winners."""
        candidate = Candidate(dspy.Predict("input -> output"))
        candidate.scores = [
            Metric(0.0, id='uuid-1'),
            Metric(0.0, id='uuid-2'),
        ]
        
        survivors = Survivors(candidate, iteration=0)
        self.selector.promote(survivors)
        
        assert candidate in self.selector.task_wins
        assert self.selector.task_wins[candidate] == 2
        assert 'uuid-1' in self.selector.example_best_candidates
        assert 'uuid-2' in self.selector.example_best_candidates

    def test_large_number_of_candidates(self):
        """Test performance with many candidates."""
        # Create 50 candidates with random scores
        candidates = []
        for i in range(50):
            scores = {j: 0.1 + (i + j) % 10 / 10.0 for j in range(5)}  # Varied scores
            candidate = self.create_candidate_with_scores(scores, generation=i)
            candidates.append(candidate)
        
        survivors = Survivors(*candidates, iteration=0)
        parents = self.selector.promote(survivors)
        
        # Should handle large numbers efficiently
        assert parents.size() > 0
        assert len(self.selector.task_wins) > 0
        assert all(wins > 0 for wins in self.selector.task_wins.values())

    def test_identical_candidates_same_scores(self):
        """Test multiple candidates with identical scores."""
        # Three candidates with exactly the same scores
        candidate_a = self.create_candidate_with_scores({0: 0.5, 1: 0.5})
        candidate_b = self.create_candidate_with_scores({0: 0.5, 1: 0.5})
        candidate_c = self.create_candidate_with_scores({0: 0.5, 1: 0.5})
        
        survivors = Survivors(candidate_a, candidate_b, candidate_c, iteration=0)
        parents = self.selector.promote(survivors)
        
        # All should coexist (no domination)
        assert parents.size() == 3
        # Each should win at least one example (or share)
        total_wins = sum(self.selector.task_wins.values())
        assert total_wins >= 2  # At least as many wins as examples

    def test_floating_point_precision_edge_cases(self):
        """Test handling of floating point precision issues."""
        # Very close scores that might cause precision issues
        candidate_a = self.create_candidate_with_scores({0: 0.7000000001})
        candidate_b = self.create_candidate_with_scores({0: 0.7000000002})
        
        survivors = Survivors(candidate_a, candidate_b, iteration=0)
        parents = self.selector.promote(survivors)
        
        # Should handle precision correctly
        assert parents.size() > 0
        # The slightly better candidate should win
        assert candidate_b in self.selector.example_best_candidates['uuid-0']

    def test_zero_and_negative_scores(self):
        """Test handling of zero and negative scores."""
        candidate_zero = self.create_candidate_with_scores({0: 0.0, 1: 0.0})
        candidate_negative = self.create_candidate_with_scores({0: -0.5, 1: -0.3})
        candidate_positive = self.create_candidate_with_scores({0: 0.1, 1: 0.2})
        
        survivors = Survivors(candidate_zero, candidate_negative, candidate_positive, iteration=0)
        parents = self.selector.promote(survivors)
        
        # Positive should dominate
        assert candidate_positive in parents.candidates
        assert candidate_positive in self.selector.example_best_candidates['uuid-0']
        assert candidate_positive in self.selector.example_best_candidates['uuid-1']


class TestParetoFrontierObserverIntegration:
    """Test observer pattern integration (simplified to avoid async complexity)."""

    def setup_method(self):
        self.selector = ParetoFrontier()
        self.selector.start_compilation(None, verbose=False)

    def create_candidate_with_scores(self, scores_dict, generation=0):
        """Helper to create candidates with UUID-based scores."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=generation)
        candidate.scores = [
            Metric(value, id=f'uuid-{task_id}', trace={'dspy_uuid': f'uuid-{task_id}'})
            for task_id, value in scores_dict.items()
        ]
        return candidate

    def test_channel_observer_structure(self):
        """Test that ParetoFrontier has Channel observer capabilities."""
        # Verify it inherits from Channel and has observer capabilities
        assert hasattr(self.selector, 'observers')
        assert hasattr(self.selector, 'publish')
        assert hasattr(self.selector, 'subscribe')
        assert hasattr(self.selector, 'unsubscribe')
        
        # Verify initial state
        assert isinstance(self.selector.observers, dict)

    def test_update_score_processing(self):
        """Test that individual score updates are processed correctly."""
        candidate = self.create_candidate_with_scores({0: 0.8})
        metric = candidate.scores[0]
        
        self.selector.update_score('uuid-0', candidate, metric)
        
        # Verify score update was processed
        assert candidate in self.selector.task_wins
        assert 'uuid-0' in self.selector.example_best_candidates

    def test_batch_score_update_processing(self):
        """Test that batch score updates are processed correctly."""
        candidate = self.create_candidate_with_scores({0: 0.8, 1: 0.6})
        survivors = Survivors(candidate, iteration=0)
        
        self.selector.update_scores_batch(survivors)
        
        # Verify batch update was processed
        assert candidate in self.selector.task_wins
        assert self.selector.task_wins[candidate] == 2
        
    def test_observer_subscription_interface(self):
        """Test observer subscription interface works."""
        # Create a simple mock observer
        mock_observer = Mock()
        mock_observer.promote = Mock()
        
        # Test subscription
        self.selector.subscribe(mock_observer, 'promote')
        
        # Verify subscription worked
        assert 'promote' in self.selector.observers
        assert len(self.selector.observers['promote']) == 1


class TestParetoFrontierBudgetIntegration:
    """Test budget integration (if applicable)."""

    def setup_method(self):
        self.selector = ParetoFrontier()
        self.selector.start_compilation(None, verbose=False)

    def create_candidate_with_scores(self, scores_dict, generation=0):
        """Helper to create candidates with UUID-based scores."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=generation)
        candidate.scores = [
            Metric(value, id=f'uuid-{task_id}', trace={'dspy_uuid': f'uuid-{task_id}'})
            for task_id, value in scores_dict.items()
        ]
        return candidate

    def test_promote_with_budget(self):
        """Test promotion with budget tracking."""
        candidate = self.create_candidate_with_scores({0: 0.8})
        survivors = Survivors(candidate, iteration=0)
        budget = LMCallsBudget(max_calls=100)
        
        parents = self.selector.promote(survivors, budget=budget)
        
        # Should handle budget parameter gracefully
        assert parents.size() == 1
        assert candidate in parents.candidates

    def test_promote_without_budget(self):
        """Test promotion without budget (default case)."""
        candidate = self.create_candidate_with_scores({0: 0.8})
        survivors = Survivors(candidate, iteration=0)
        
        parents = self.selector.promote(survivors)  # No budget
        
        assert parents.size() == 1
        assert candidate in parents.candidates


class TestParetoFrontierAlgorithmCompliance:
    """Test compliance with GEPA Algorithm 2 specification."""

    def setup_method(self):
        self.selector = ParetoFrontier()
        self.selector.start_compilation(None, verbose=False)

    def create_candidate_with_scores(self, scores_dict, generation=0):
        """Helper to create candidates with UUID-based scores."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=generation)
        candidate.scores = [
            Metric(value, id=f'uuid-{task_id}', trace={'dspy_uuid': f'uuid-{task_id}'})
            for task_id, value in scores_dict.items()
        ]
        return candidate

    def test_algorithm_2_step_1_accumulate_task_winners(self):
        """Test Algorithm 2 Step 1: Accumulate candidates that win at least one task."""
        # Candidates with different strengths
        candidate_a = self.create_candidate_with_scores({0: 0.9, 1: 0.2, 2: 0.3})  # Wins task 0
        candidate_b = self.create_candidate_with_scores({0: 0.3, 1: 0.8, 2: 0.2})  # Wins task 1
        candidate_c = self.create_candidate_with_scores({0: 0.1, 1: 0.1, 2: 0.95}) # Wins task 2
        candidate_d = self.create_candidate_with_scores({0: 0.1, 1: 0.1, 2: 0.1})  # Wins nothing
        
        survivors = Survivors(candidate_a, candidate_b, candidate_c, candidate_d, iteration=0)
        parents = self.selector.promote(survivors)
        
        # Step 1: Only task winners should be accumulated
        assert candidate_a in self.selector.task_wins  # Wins task 0
        assert candidate_b in self.selector.task_wins  # Wins task 1
        assert candidate_c in self.selector.task_wins  # Wins task 2
        # candidate_d wins no tasks, but might still be included if not dominated

    def test_algorithm_2_step_2_remove_dominated(self):
        """Test Algorithm 2 Step 2: Remove strictly dominated candidates."""
        # Create domination scenario
        dominant = self.create_candidate_with_scores({0: 0.9, 1: 0.8, 2: 0.7})  # Superior on all
        dominated = self.create_candidate_with_scores({0: 0.3, 1: 0.2, 2: 0.1})  # Inferior on all
        
        assert dominant.dominate(dominated)  # Verify domination
        
        survivors = Survivors(dominant, dominated, iteration=0)
        parents = self.selector.promote(survivors)
        
        # Step 2: Dominated candidates should be removed
        assert dominant in parents.candidates
        assert dominated not in parents.candidates
        assert dominant in self.selector.task_wins
        assert dominated not in self.selector.task_wins

    def test_algorithm_2_pareto_frontier_property(self):
        """Test that result satisfies Pareto frontier property."""
        # Create complex scenario with multiple candidates
        candidates = [
            self.create_candidate_with_scores({0: 0.9, 1: 0.3, 2: 0.2}),  # Good at 0
            self.create_candidate_with_scores({0: 0.2, 1: 0.9, 2: 0.3}),  # Good at 1
            self.create_candidate_with_scores({0: 0.3, 1: 0.2, 2: 0.9}),  # Good at 2
            self.create_candidate_with_scores({0: 0.6, 1: 0.6, 2: 0.6}),  # Balanced
            self.create_candidate_with_scores({0: 0.1, 1: 0.1, 2: 0.1}),  # Poor (should be dominated)
        ]
        
        survivors = Survivors(*candidates, iteration=0)
        parents = self.selector.promote(survivors)
        
        # Verify Pareto frontier property: no candidate dominates another in the result
        parent_list = list(parents.candidates)
        for i, candidate_a in enumerate(parent_list):
            for j, candidate_b in enumerate(parent_list):
                if i != j:
                    assert not candidate_a.dominate(candidate_b), f"Candidate {i} dominates candidate {j} in Pareto frontier"

    def test_algorithm_2_incremental_updates(self):
        """Test that algorithm works correctly with incremental candidate additions."""
        # Start with one candidate
        candidate_1 = self.create_candidate_with_scores({0: 0.5, 1: 0.5})
        survivors_1 = Survivors(candidate_1, iteration=0)
        parents_1 = self.selector.promote(survivors_1)
        
        assert parents_1.size() == 1
        initial_wins = self.selector.task_wins[candidate_1]
        
        # Add better candidate for one task
        candidate_2 = self.create_candidate_with_scores({0: 0.9, 1: 0.2})
        survivors_2 = Survivors(candidate_2, iteration=1)
        parents_2 = self.selector.promote(survivors_2)
        
        # Verify incremental update worked correctly
        assert candidate_2 in self.selector.example_best_candidates['uuid-0']  # Took over task 0
        assert candidate_1 in self.selector.example_best_candidates['uuid-1']  # Still owns task 1
        assert self.selector.task_wins[candidate_1] < initial_wins  # Lost some wins


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
