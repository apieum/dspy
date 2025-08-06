"""Ancestry traversal utilities for GEPA genealogy operations."""

from abc import ABC, abstractmethod
from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.candidate import Candidate


class AncestryTraverser(ABC):
    """Base class for traversing candidate genealogy.
    
    Allows different traversal behaviors and enables parallel processing
    of genealogy operations without modifying Candidate class.
    """
    
    @abstractmethod
    def process_candidate(self, candidate: "Candidate") -> None:
        """Process a single candidate. Override to define behavior.
        
        Args:
            candidate: The candidate being processed
        """
        pass
    
    def traverse(self, candidate: "Candidate") -> None:
        """Traverse candidate and all ancestors.
        
        Uses depth-first traversal with cycle detection.
        Can be extended for parallel processing of genealogy trees.
        
        Args:
            candidate: Starting candidate for traversal
        """
        visited = set()
        self._traverse_recursive(candidate, visited)
    
    def _traverse_recursive(self, candidate: "Candidate", visited: Set["Candidate"]) -> None:
        """Recursive helper for traversal with cycle detection.
        
        Args:
            candidate: Current candidate to process
            visited: Set of already visited candidates (cycle detection)
        """
        if candidate in visited:
            return  # Prevent infinite loops in genealogy
        
        visited.add(candidate)
        self.process_candidate(candidate)
        
        # Process all parents recursively (can be parallelized)
        if candidate.parents:
            for parent in candidate.parents:
                self._traverse_recursive(parent, visited)


class CommonAncestorFinder(AncestryTraverser):
    """Find common ancestors between multiple candidates."""
    
    def __init__(self):
        self.ancestor_sets = []
        self.common_ancestors: Set["Candidate"] = set()
    
    def process_candidate(self, candidate: "Candidate") -> None:
        """Add candidate to current ancestor set."""
        if hasattr(self, '_current_set'):
            self._current_set.add(candidate)
    
    def find_common_ancestors(self, candidates: list["Candidate"]) -> Set["Candidate"]:
        """Find ancestors common to all provided candidates.
        
        Args:
            candidates: List of candidates to find common ancestors for
            
        Returns:
            Set of candidates that are ancestors to all input candidates
        """
        if not candidates:
            return set()
        
        # Collect ancestors for each candidate (can be parallelized)
        self.ancestor_sets = []
        for candidate in candidates:
            self._current_set = set()
            self.traverse(candidate)
            # Remove the candidate itself from its ancestor set
            self._current_set.discard(candidate)
            self.ancestor_sets.append(self._current_set)
        
        # Find intersection of all ancestor sets
        if self.ancestor_sets:
            self.common_ancestors = self.ancestor_sets[0].copy()
            for ancestor_set in self.ancestor_sets[1:]:
                self.common_ancestors.intersection_update(ancestor_set)
        else:
            self.common_ancestors = set()
        
        return self.common_ancestors


class AncestryChecker(AncestryTraverser):
    """Check ancestry relationships between candidates."""
    
    def __init__(self, potential_ancestor: "Candidate"):
        self.potential_ancestor = potential_ancestor
        self.is_ancestor = False
    
    def process_candidate(self, candidate: "Candidate") -> None:
        """Check if processed candidate is the potential ancestor."""
        if candidate is self.potential_ancestor:
            self.is_ancestor = True


# Direct functions for the two operations we actually need
def find_common_ancestors(candidate1: "Candidate", candidate2: "Candidate") -> Set["Candidate"]:
    """Find common ancestors between two candidates.
    
    Args:
        candidate1: First candidate
        candidate2: Second candidate
        
    Returns:
        Set of candidates that are ancestors to both inputs
    """
    finder = CommonAncestorFinder()
    return finder.find_common_ancestors([candidate1, candidate2])


def is_ancestor_of(potential_ancestor: "Candidate", descendant: "Candidate") -> bool:
    """Check if one candidate is an ancestor of another.
    
    Args:
        potential_ancestor: Candidate that might be an ancestor
        descendant: Candidate that might be a descendant
        
    Returns:
        True if potential_ancestor is an ancestor of descendant
    """
    checker = AncestryChecker(potential_ancestor)
    checker.traverse(descendant)
    return checker.is_ancestor