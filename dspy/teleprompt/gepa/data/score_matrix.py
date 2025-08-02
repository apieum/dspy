"""ScoreMatrix for tracking best candidates per task."""

from typing import Dict, List, Optional, Any

# Import Candidate with relative import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .candidate import Candidate


class ScoreMatrix:
    """Task-indexed matrix storing best candidates per task.
    
    Maintains weak references to the best performing candidate for each task.
    """
    
    def __init__(self):
        import weakref
        # task_id -> weakref to best candidate for that task
        self.best_candidates: Dict[int, Any] = {}  # weakref.ref objects
        
    def update_scores(self, candidates: List['Candidate']) -> None:
        """Update the score matrix with new candidates.
        
        For each candidate, check if it's better than current best for any task.
        If so, store a weak reference to the candidate.
        """
        import weakref
        
        for candidate in candidates:
            # Iterate over candidate's task scores
            for task_id, score in enumerate(candidate.task_scores):
                if score > 0:  # Only consider non-zero scores
                    self._maybe_update_best_for_task(task_id, candidate, score)
    
    def _maybe_update_best_for_task(self, task_id: int, candidate: 'Candidate', score: float) -> None:
        """Update best candidate for task if this candidate is better."""
        import weakref
        
        current_best_ref = self.best_candidates.get(task_id)
        
        if current_best_ref is None:
            # No current best - this candidate becomes best
            self.best_candidates[task_id] = weakref.ref(candidate)
        else:
            # Check if current best still exists
            current_best = current_best_ref()
            if current_best is None:
                # Current best was garbage collected - use new candidate
                self.best_candidates[task_id] = weakref.ref(candidate)
            else:
                # Compare scores
                current_score = current_best.get_task_score(task_id) or 0.0
                
                # Replace if new candidate has better score, or same score but more recent generation
                if (score > current_score or 
                    (score == current_score and candidate.generation_number > current_best.generation_number)):
                    self.best_candidates[task_id] = weakref.ref(candidate)
    
    def get_best_candidate_for_task(self, task_id: int) -> Optional['Candidate']:
        """Get the single best candidate for a specific task."""
        best_ref = self.best_candidates.get(task_id)
        if best_ref is None:
            return None
        
        best_candidate = best_ref()  # Dereference weakref
        return best_candidate  # May be None if garbage collected
    
    def get_all_task_ids(self) -> List[int]:
        """Get all task IDs in the matrix."""
        return list(self.best_candidates.keys())