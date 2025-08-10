"""Test cohort stochastic sampling functionality."""

import pytest
import random
from unittest.mock import Mock

from dspy.teleprompt.darwin.data.cohort import Cohort, Parents, Survivors, NewBorns
from dspy.teleprompt.darwin.data.candidate import Candidate


class MockCandidate:
    """Simple mock candidate for testing."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"MockCandidate({self.name})"

    def __eq__(self, other):
        return isinstance(other, MockCandidate) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class TestCohortFeatureExtraction:
    """Test feature value extraction from different specification types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = MockCandidate("a")
        self.b = MockCandidate("b")
        self.c = MockCandidate("c")
        self.candidates = [self.a, self.b, self.c]

    def test_dict_feature_extraction(self):
        """Test extraction from dict mapping candidate -> value."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 1, self.c: 0})

        assert cohort.weights['task_wins'][self.a] == 10.0
        assert cohort.weights['task_wins'][self.b] == 1.0
        assert cohort.weights['task_wins'][self.c] == 0.0

    def test_dict_feature_missing_candidate(self):
        """Test dict extraction with missing candidate uses 0.0."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10})

        assert cohort.weights['task_wins'][self.a] == 10.0
        assert cohort.weights['task_wins'][self.b] == 0.0
        assert cohort.weights['task_wins'][self.c] == 0.0

    def test_callable_feature_extraction(self):
        """Test extraction from callable function."""
        def score_func(candidate):
            scores = {self.a: 0.8, self.b: 0.5, self.c: 0.2}
            return scores.get(candidate, 0.0)

        cohort = Cohort(self.a, self.b, self.c, average_score=score_func)

        assert cohort.weights['average_score'][self.a] == 0.8
        assert cohort.weights['average_score'][self.b] == 0.5
        assert cohort.weights['average_score'][self.c] == 0.2

    def test_sequence_feature_extraction(self):
        """Test extraction from list/tuple sequence."""
        cohort = Cohort(self.a, self.b, self.c, scores=[0.9, 0.4, 0.1])

        # Note: order depends on set iteration, so we check all values are present
        values = list(cohort.weights['scores'].values())
        assert sorted(values) == [0.1, 0.4, 0.9]

    def test_sequence_feature_wrong_length(self):
        """Test sequence with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Sequence feature has length 2 but cohort has 3 candidates"):
            Cohort(self.a, self.b, self.c, scores=[0.9, 0.4])  # Too short

    def test_scalar_feature_extraction(self):
        """Test extraction from scalar numeric (global bias)."""
        cohort = Cohort(self.a, self.b, self.c, bias=0.5)

        assert cohort.weights['bias'][self.a] == 0.5
        assert cohort.weights['bias'][self.b] == 0.5
        assert cohort.weights['bias'][self.c] == 0.5

    def test_unsupported_feature_type(self):
        """Test unsupported feature type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported feature specification type"):
            Cohort(self.a, self.b, self.c, bad_feature="string")

    def test_multiple_features(self):
        """Test cohort with multiple features."""
        cohort = Cohort(
            self.a, self.b, self.c,
            task_wins={self.a: 10, self.b: 1, self.c: 0},
            average_score={self.a: 0.8, self.b: 0.5, self.c: 0.2},
            bias=0.1
        )

        assert 'task_wins' in cohort.weights
        assert 'average_score' in cohort.weights
        assert 'bias' in cohort.weights
        assert len(cohort.weights) == 3


class TestCohortStochasticSampling:
    """Test stochastic sampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = MockCandidate("a")
        self.b = MockCandidate("b")
        self.c = MockCandidate("c")

    def test_deterministic_single_selection(self):
        """Test that heavily weighted candidate is selected deterministically."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 1, self.c: 0})

        # With deterministic RNG, should consistently pick the same candidate
        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=1, rng=rng)
        first_result = sample.first()

        # Test multiple samples with same seed - should be consistent
        for _ in range(10):
            rng = random.Random(42)  # Reset seed
            sample = cohort.sample_stochastic(n=1, rng=rng)
            assert sample.first() == first_result

        # With these weights (10:1:0), the heavily weighted candidate should be selected
        assert first_result == self.a

    def test_weighted_combination_explicit_factors(self):
        """Test weighted combination with explicit factor multipliers."""
        cohort = Cohort(
            self.a, self.b, self.c,
            task_wins={self.a: 2, self.b: 1, self.c: 0},
            average_score={self.a: 0.5, self.b: 0.8, self.c: 0.3}
        )

        # Give more weight to average_score (2.0) than task_wins (1.0)
        # Candidate weights:
        # a: 2*1.0 + 0.5*2.0 = 3.0
        # b: 1*1.0 + 0.8*2.0 = 2.6
        # c: 0*1.0 + 0.3*2.0 = 0.6

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=1, rng=rng, task_wins=1.0, average_score=2.0)

        # With deterministic seed, result should be consistent
        first_result = sample.first()

        # Test multiple samples with same seed - should be consistent
        for _ in range(5):
            rng = random.Random(42)  # Reset seed
            sample = cohort.sample_stochastic(n=1, rng=rng, task_wins=1.0, average_score=2.0)
            assert sample.first() == first_result

        # Verify it's not candidate c (lowest weight)
        assert first_result != self.c

    def test_default_behavior_uses_all_features(self):
        """Test default behavior uses all features with equal weight 1.0."""
        cohort = Cohort(
            self.a, self.b, self.c,
            task_wins={self.a: 3, self.b: 1, self.c: 0},
            average_score={self.a: 0.8, self.b: 0.5, self.c: 0.2}
        )

        # Default should use both features with weight 1.0
        # Candidate weights:
        # a: 3*1.0 + 0.8*1.0 = 3.8
        # b: 1*1.0 + 0.5*1.0 = 1.5
        # c: 0*1.0 + 0.2*1.0 = 0.2

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=1, rng=rng)  # No explicit factors, should use defaults

        # With highest combined weight, 'a' should be selected
        assert sample.first() == self.a

    # @pytest.mark.slow_test
    def test_fallback_uniform_sampling(self):
        """Test uniform sampling when all weights are zero."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 0, self.b: 0, self.c: 0})

        # With all zero weights, should fall back to uniform
        results = set()
        for seed in range(30):  # sufficient for statistical validation
            rng = random.Random(seed)
            sample = cohort.sample_stochastic(n=1, rng=rng)
            results.add(sample.first())

        # Should see all candidates selected across different seeds
        assert len(results) == 3  # All candidates should appear
        assert self.a in results
        assert self.b in results
        assert self.c in results

    # @pytest.mark.slow_test
    def test_no_features_uniform_sampling(self):
        """Test uniform sampling when no features provided."""
        cohort = Cohort(self.a, self.b, self.c)  # No features

        results = set()
        for seed in range(30):  # sufficient for statistical validation
            rng = random.Random(seed)
            sample = cohort.sample_stochastic(n=1, rng=rng)
            results.add(sample.first())

        # Should see all candidates selected across different seeds
        assert len(results) == 3

    def test_multiple_candidate_sampling(self):
        """Test sampling multiple candidates."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=2, rng=rng)

        # With random.choices, we can get duplicates, so size might be 1 or 2
        assert sample.size() >= 1
        assert sample.size() <= 2
        # All selected candidates should be from original cohort
        for candidate in sample.candidates:
            assert candidate in [self.a, self.b, self.c]

    def test_sample_more_than_available(self):
        """Test sampling more candidates than available returns all."""
        cohort = Cohort(self.a, self.b, task_wins={self.a: 10, self.b: 5})

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=5, rng=rng)  # More than 2 available
        assert sample.size() == 2
        assert self.a in sample.candidates
        assert self.b in sample.candidates

    def test_exclude_parameter(self):
        """Test exclude parameter filters out candidates."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=1, exclude=[self.a], rng=rng)
        assert sample.first() in [self.b, self.c]
        assert self.a not in sample.candidates

    def test_return_type_preservation(self):
        """Test that sample_stochastic returns same cohort type."""
        rng = random.Random(42)

        parents = Parents(self.a, self.b, task_wins={self.a: 10, self.b: 5})
        sample = parents.sample_stochastic(n=1, rng=rng)
        assert isinstance(sample, Parents)

        survivors = Survivors(self.a, self.b, task_wins={self.a: 10, self.b: 5})
        sample = survivors.sample_stochastic(n=1, rng=rng)
        assert isinstance(sample, Survivors)

    def test_weight_preservation_in_sample(self):
        """Test that weights are preserved in sampled cohorts."""
        cohort = Cohort(
            self.a, self.b, self.c,
            task_wins={self.a: 10, self.b: 5, self.c: 1},
            scores={self.a: 0.8, self.b: 0.6, self.c: 0.4}
        )

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=2, rng=rng)

        # Sampled cohort should have same features
        assert 'task_wins' in sample.weights
        assert 'scores' in sample.weights

        # But only for sampled candidates
        for candidate in sample.candidates:
            assert candidate in sample.weights['task_wins']
            assert candidate in sample.weights['scores']


class TestCohortFeatureManagement:
    """Test add_feature and del_feature functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = MockCandidate("a")
        self.b = MockCandidate("b")
        self.c = MockCandidate("c")

    def test_add_feature_after_init(self):
        """Test adding features after cohort initialization."""
        cohort = Cohort(self.a, self.b, self.c)

        cohort.add_feature('new_feature', {self.a: 10, self.b: 5, self.c: 1})

        assert 'new_feature' in cohort.weights
        assert cohort.weights['new_feature'][self.a] == 10.0
        assert cohort.weights['new_feature'][self.b] == 5.0
        assert cohort.weights['new_feature'][self.c] == 1.0

    def test_delete_feature(self):
        """Test removing features."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        assert 'task_wins' in cohort.weights
        cohort.del_feature('task_wins')
        assert 'task_wins' not in cohort.weights

    def test_delete_nonexistent_feature(self):
        """Test deleting non-existent feature doesn't error."""
        cohort = Cohort(self.a, self.b, self.c)

        # Should not raise error
        cohort.del_feature('nonexistent')

    def test_attribute_access_to_features(self):
        """Test accessing features via attribute notation."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        # Should be able to access feature via attribute
        task_wins = cohort.task_wins
        assert task_wins == cohort.weights['task_wins']

    def test_attribute_error_for_missing_feature(self):
        """Test AttributeError for missing feature."""
        cohort = Cohort(self.a, self.b, self.c)

        with pytest.raises(AttributeError, match="'Cohort' object has no attribute 'nonexistent'"):
            _ = cohort.nonexistent


class TestCohortSamplingEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = MockCandidate("a")
        self.b = MockCandidate("b")
        self.c = MockCandidate("c")

    def test_empty_cohort_sampling(self):
        """Test sampling from empty cohort."""
        cohort = Cohort()

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=1, rng=rng)
        assert sample.is_empty()

    def test_single_candidate_sampling(self):
        """Test sampling from single-candidate cohort."""
        cohort = Cohort(self.a, task_wins={self.a: 10})

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=1, rng=rng)
        assert sample.first() == self.a

    def test_zero_sample_size(self):
        """Test sampling zero candidates."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=0, rng=rng)
        assert sample.is_empty()

    def test_factor_for_missing_feature(self):
        """Test providing factor for non-existent feature is ignored."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        rng = random.Random(42)
        # Should not error, nonexistent factor should be ignored
        sample = cohort.sample_stochastic(n=1, rng=rng, task_wins=1.0, nonexistent=2.0)
        assert not sample.is_empty()

    # @pytest.mark.slow_test
    def test_negative_factors_handling(self):
        """Test that negative factors are handled gracefully."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        # Your implementation uses abs() to handle negative factors
        # Negative factors should be treated as 0, leading to uniform sampling
        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=1, rng=rng, task_wins=-1.0)
        assert not sample.is_empty()

        # With negative factor, all weights become 0, should fall back to uniform
        results = []
        for seed in range(20):  # sufficient for statistical validation
            rng = random.Random(seed)
            sample = cohort.sample_stochastic(n=1, rng=rng, task_wins=-1.0)
            results.append(sample.first())

        # Should see all candidates roughly equally (uniform distribution)
        unique_results = set(results)
        assert len(unique_results) >= 2  # Should see multiple candidates


class TestParentsBackwardCompatibility:
    """Test Parents class backward compatibility."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = MockCandidate("a")
        self.b = MockCandidate("b")
        self.c = MockCandidate("c")

    def test_parents_with_task_wins(self):
        """Test Parents behaves like old implementation."""
        parents = Parents(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 1, self.c: 0})

        # Should have task_wins feature
        assert 'task_wins' in parents.weights

        # Sampling should work
        rng = random.Random(42)
        sample = parents.sample_stochastic(n=1, rng=rng)
        assert isinstance(sample, Parents)
        assert not sample.is_empty()

    # @pytest.mark.slow_test
    def test_parents_delegation_behavior(self):
        """Test that Parents delegates to Cohort.sample_stochastic."""
        parents = Parents(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 1, self.c: 0})

        # Create equivalent regular cohort
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 1, self.c: 0})

        # Both should behave similarly
        parents_results = []
        cohort_results = []

        for seed in range(30):  # sufficient for statistical validation
            rng_parents = random.Random(seed)
            rng_cohort = random.Random(seed)
            parents_sample = parents.sample_stochastic(n=1, rng=rng_parents)
            cohort_sample = cohort.sample_stochastic(n=1, rng=rng_cohort)
            parents_results.append(parents_sample.first())
            cohort_results.append(cohort_sample.first())

        # Both should have same weights structure
        rng_parents = random.Random(42)
        rng_cohort = random.Random(42)
        parents_sample = parents.sample_stochastic(n=1, rng=rng_parents)
        cohort_sample = cohort.sample_stochastic(n=1, rng=rng_cohort)
        assert parents_sample.weights.keys() == cohort_sample.weights.keys()

        # Both should show similar bias toward heavily weighted candidate
        parents_a_count = parents_results.count(self.a)
        cohort_a_count = cohort_results.count(self.a)
        assert parents_a_count > 24  # Both should heavily favor 'a'
        assert cohort_a_count > 24


class TestCohortSamplingPerformance:
    """Test performance characteristics and edge cases."""

     # @pytest.mark.slow_test
    def test_large_cohort_sampling(self):
        """Test sampling from large cohort doesn't fail."""
        # Create moderately large cohort
        candidates = [MockCandidate(f"candidate_{i}") for i in range(50)]
        weights = {c: random.random() for c in candidates}

        cohort = Cohort(*candidates, task_wins=weights)

        # Should handle large sampling efficiently
        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=10, rng=rng)
        assert sample.size() <= 10
        assert all(c in cohort.candidates for c in sample.candidates)

    # @pytest.mark.slow_test
    def test_many_features_sampling(self):
        """Test cohort with many features."""
        candidates = [MockCandidate(f"candidate_{i}") for i in range(10)]

        # Create cohort with multiple features
        feature_kwargs = {}
        for i in range(10):
            feature_kwargs[f'feature_{i}'] = {c: random.random() for c in candidates}

        cohort = Cohort(*candidates, **feature_kwargs)

        # Should handle many features
        rng = random.Random(42)
        sample = cohort.sample_stochastic(n=3, rng=rng)
        assert sample.size() <= 3
        assert len(sample.weights) == 10  # All features preserved


class TestCohortInternalHelpers:
    """Test internal helper methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = MockCandidate("a")
        self.b = MockCandidate("b")
        self.c = MockCandidate("c")

    def test_new_weights_helper(self):
        """Test _new_weights helper method."""
        cohort = Cohort(
            self.a, self.b, self.c,
            task_wins={self.a: 10, self.b: 5, self.c: 1},
            scores={self.a: 0.8, self.b: 0.6, self.c: 0.4}
        )

        # Test with subset of candidates
        new_weights = cohort._new_weights([self.a, self.b])

        assert 'task_wins' in new_weights
        assert 'scores' in new_weights
        assert self.a in new_weights['task_wins']
        assert self.b in new_weights['task_wins']
        assert self.c not in new_weights['task_wins']

        # Values should be preserved
        assert new_weights['task_wins'][self.a] == 10.0
        assert new_weights['task_wins'][self.b] == 5.0

    def test_new_weights_empty_list(self):
        """Test _new_weights with empty candidate list."""
        cohort = Cohort(self.a, self.b, self.c, task_wins={self.a: 10, self.b: 5, self.c: 1})

        new_weights = cohort._new_weights([])
        assert 'task_wins' in new_weights
        assert len(new_weights['task_wins']) == 0
