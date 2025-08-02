"""Generation data structure for GEPA optimization."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

import dspy

# Import Candidate with relative import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..data.candidate import Candidate


@dataclass
class Generation:
    """A generation of candidates created in a single iteration.
    
    Represents a cohort of candidates that were generated together
    and share the same generation context.
    """
    candidates: List['Candidate']
    generation_id: int = 0
    iteration: int = 0
    creation_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)  # For generation-specific info
    
    def is_empty(self) -> bool:
        """Check if generation has no candidates."""
        return len(self.candidates) == 0
    
    def size(self) -> int:
        """Number of candidates in this generation."""
        return len(self.candidates)
    
    def add_candidate(self, candidate: 'Candidate') -> None:
        """Add a candidate to this generation."""
        candidate.generation_number = self.generation_id
        self.candidates.append(candidate)
    
    def get_best_candidate(self, metric: Callable, examples: List[dspy.Example]) -> Optional['Candidate']:
        """Get the best performing candidate in this generation."""
        if not self.candidates:
            return None
        
        best_candidate = None
        best_score = -float('inf')
        
        for candidate in self.candidates:
            score = candidate.evaluate_on_batch(examples, metric)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                
        return best_candidate