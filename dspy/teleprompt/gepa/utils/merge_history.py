"""Merge history tracking for System-Aware Merge algorithm."""

import logging
from typing import Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..data.candidate import Candidate

logger = logging.getLogger(__name__)


@dataclass
class MergeAttempt:
    """Record of a merge attempt between candidates.
    
    Tracks the specific combination of parent1, parent2, and ancestor
    that was attempted, along with the outcome.
    """
    parent1_id: int
    parent2_id: int 
    ancestor_id: int
    iteration: int
    successful: bool = False
    failure_reason: str = ""
    
    def __post_init__(self):
        """Ensure consistent ordering of parents for deduplication."""
        # Always store parents in consistent order (smaller id first)
        if self.parent1_id > self.parent2_id:
            self.parent1_id, self.parent2_id = self.parent2_id, self.parent1_id
    
    @property
    def merge_key(self) -> Tuple[int, int, int]:
        """Get the unique key for this merge combination."""
        return (self.parent1_id, self.parent2_id, self.ancestor_id)


class MergeHistoryTracker:
    """Tracks attempted merge combinations to prevent redundant attempts.
    
    As specified in Algorithm 4, we need to check if a specific
    (parent1, parent2, ancestor) combination has been tried before.
    """
    
    def __init__(self):
        self.attempted_merges: Set[Tuple[int, int, int]] = set()
        self.merge_attempts: list[MergeAttempt] = []
        self._candidate_id_counter = 0
        self._candidate_to_id = {}
    
    def _get_candidate_id(self, candidate: "Candidate") -> int:
        """Get or assign a unique ID for a candidate.
        
        Uses object identity (id()) as the basis for candidate identification.
        
        Args:
            candidate: Candidate to get ID for
            
        Returns:
            Unique integer ID for the candidate
        """
        candidate_key = id(candidate)
        if candidate_key not in self._candidate_to_id:
            self._candidate_to_id[candidate_key] = self._candidate_id_counter
            self._candidate_id_counter += 1
        return self._candidate_to_id[candidate_key]
    
    def has_been_attempted(self, parent1: "Candidate", parent2: "Candidate", ancestor: "Candidate") -> bool:
        """Check if this specific merge combination has been attempted before.
        
        Args:
            parent1: First parent candidate
            parent2: Second parent candidate  
            ancestor: Ancestor candidate to use as base
            
        Returns:
            True if this combination has been attempted before
        """
        try:
            parent1_id = self._get_candidate_id(parent1)
            parent2_id = self._get_candidate_id(parent2)
            ancestor_id = self._get_candidate_id(ancestor)
            
            # Create merge attempt to get consistent ordering
            attempt = MergeAttempt(parent1_id, parent2_id, ancestor_id, iteration=0)
            merge_key = attempt.merge_key
            
            return merge_key in self.attempted_merges
            
        except Exception as e:
            logger.warning(f"Error checking merge history: {e}")
            return False  # Default to allowing attempt
    
    def record_attempt(self, parent1: "Candidate", parent2: "Candidate", ancestor: "Candidate", 
                      iteration: int, successful: bool = False, failure_reason: str = "") -> None:
        """Record a merge attempt for future reference.
        
        Args:
            parent1: First parent candidate
            parent2: Second parent candidate
            ancestor: Ancestor candidate used as base
            iteration: Current iteration number
            successful: Whether the merge was successful
            failure_reason: Reason for failure (if unsuccessful)
        """
        try:
            parent1_id = self._get_candidate_id(parent1)
            parent2_id = self._get_candidate_id(parent2)
            ancestor_id = self._get_candidate_id(ancestor)
            
            # Create and record the attempt
            attempt = MergeAttempt(
                parent1_id=parent1_id,
                parent2_id=parent2_id,
                ancestor_id=ancestor_id,
                iteration=iteration,
                successful=successful,
                failure_reason=failure_reason
            )
            
            # Add to attempted merges set
            self.attempted_merges.add(attempt.merge_key)
            
            # Add to full history
            self.merge_attempts.append(attempt)
            
            logger.debug(f"Recorded merge attempt: {attempt.merge_key} "
                        f"(successful: {successful}, reason: {failure_reason})")
            
        except Exception as e:
            logger.warning(f"Error recording merge attempt: {e}")
    
    def get_successful_merges(self) -> list[MergeAttempt]:
        """Get all successful merge attempts.
        
        Returns:
            List of MergeAttempt objects that were successful
        """
        return [attempt for attempt in self.merge_attempts if attempt.successful]
    
    def get_failed_merges(self) -> list[MergeAttempt]:
        """Get all failed merge attempts.
        
        Returns:
            List of MergeAttempt objects that failed
        """
        return [attempt for attempt in self.merge_attempts if not attempt.successful]
    
    def get_merge_statistics(self) -> dict:
        """Get statistics about merge attempts.
        
        Returns:
            Dictionary with merge attempt statistics
        """
        total_attempts = len(self.merge_attempts)
        successful_attempts = len(self.get_successful_merges())
        failed_attempts = len(self.get_failed_merges())
        
        success_rate = (successful_attempts / total_attempts) if total_attempts > 0 else 0.0
        
        # Count failure reasons
        failure_reasons = {}
        for attempt in self.get_failed_merges():
            reason = attempt.failure_reason or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "failed_attempts": failed_attempts,
            "success_rate": success_rate,
            "failure_reasons": failure_reasons,
            "unique_combinations_attempted": len(self.attempted_merges)
        }
    
    def clear_history(self) -> None:
        """Clear all merge history (useful for testing or reset)."""
        self.attempted_merges.clear()
        self.merge_attempts.clear()
        self._candidate_id_counter = 0
        self._candidate_to_id.clear()
        logger.debug("Cleared merge history")
    
    def __len__(self) -> int:
        """Return number of unique merge combinations attempted."""
        return len(self.attempted_merges)