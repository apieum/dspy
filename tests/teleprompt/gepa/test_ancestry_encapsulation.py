"""Tests for ancestry encapsulation in SystemAwareMerge.

This test demonstrates the improved design with proper encapsulation
of ancestry operations within the Candidate class.
"""

import pytest
from unittest.mock import Mock

import dspy
from dspy.teleprompt.gepa.data.candidate import Candidate
from dspy.teleprompt.gepa.generation.crossover import SystemAwareMerge


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
