"""Cohort data structure for GEPA optimization."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

import dspy

# Import Candidate with relative import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..data.candidate import Candidate


@dataclass
class Cohort:
    """A cohort of candidates created in a single iteration.
    
    Represents a group of candidates that were generated together
    and share the same iteration context. The iteration_id is stored
    in the candidates themselves as they're always from the same iteration.
    """
    candidates: List['Candidate']
    creation_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)  # For cohort-specific info
    
    @property
    def iteration_id(self) -> int:
        """Get iteration ID from candidates (they're all from same iteration)."""
        if not self.candidates:
            return 0
        return self.candidates[0].generation_number
    
    def is_empty(self) -> bool:
        """Check if cohort has no candidates."""
        return len(self.candidates) == 0
    
    def size(self) -> int:
        """Number of candidates in this cohort."""
        return len(self.candidates)
    
    def add_candidate(self, candidate: 'Candidate') -> None:
        """Add a candidate to this cohort."""
        # Note: candidate.generation_number should already be set
        self.candidates.append(candidate)
    
    def append(self, task_id: int, candidate: 'Candidate') -> None:
        """Add a candidate to this cohort (protocol method for accumulator pattern).
        
        Args:
            task_id: The task this candidate won
            candidate: The winning candidate for this task
        """
        self.add_candidate(candidate)


@dataclass
class FilteredCohort(Cohort):
    """Cohort after evaluation filtering with scores compiled."""
    filtered_count: int = 0  # How many were filtered out
    evaluation_summary: str = ""  # Summary of evaluation results