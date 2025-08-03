"""ScoreMatrix for tracking best candidates per task."""

from typing import Dict, List, Optional, Any

# Import Candidate and Cohort with relative import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .candidate import Candidate
    from .cohort import Cohort


class ScoreMatrix:
    """Task-indexed matrix storing best candidates per task.
    
    Optimized structure:
    - candidates: List containing only candidates that perform best on at least one task
    - task_scores: List where position corresponds to task_id, value is index into candidates list
    
    This allows efficient memory usage and tracking of candidates that excel in multiple tasks.
    """
    
    def __init__(self):
        # Candidates that perform best on at least one task
        self.candidates: List['Candidate'] = []
        
        # task_id -> index in candidates list (or -1 if no best candidate for task)
        self.task_scores: List[int] = []
        
    def update_scores(self, cohort: 'Cohort') -> None:
        """Update the score matrix with new candidates from a cohort.
        
        For each candidate in the cohort, check if it's better than current best for any task.
        If so, update the candidates list and task_scores mapping.
        """
        for candidate in cohort.candidates:
            # Iterate over candidate's task scores
            for task_id, score in enumerate(candidate.task_scores):
                if score > 0:  # Only consider non-zero scores
                    self._maybe_update_best_for_task(task_id, candidate, score)
    
    def _maybe_update_best_for_task(self, task_id: int, candidate: 'Candidate', score: float) -> None:
        """Update best candidate for task if this candidate is better."""
        # Ensure task_scores list is large enough
        while len(self.task_scores) <= task_id:
            self.task_scores.append(-1)  # -1 means no best candidate for this task
        
        current_best_idx = self.task_scores[task_id]
        
        if current_best_idx == -1:
            # No current best - add this candidate
            candidate_idx = self._add_or_find_candidate(candidate)
            self.task_scores[task_id] = candidate_idx
        else:
            # Compare with current best
            current_best = self.candidates[current_best_idx]
            current_score = current_best.task_score(task_id) or 0.0
            
            # Replace if new candidate has better score, or same score but more recent generation
            if (score > current_score or 
                (score == current_score and candidate.generation_number > current_best.generation_number)):
                candidate_idx = self._add_or_find_candidate(candidate)
                self.task_scores[task_id] = candidate_idx
    
    def get_best_candidate_for_task(self, task_id: int) -> Optional['Candidate']:
        """Get the single best candidate for a specific task."""
        if task_id >= len(self.task_scores) or self.task_scores[task_id] == -1:
            return None
        
        candidate_idx = self.task_scores[task_id]
        return self.candidates[candidate_idx]
    
    def get_all_task_ids(self) -> List[int]:
        """Get all task IDs in the matrix."""
        return [i for i, candidate_idx in enumerate(self.task_scores) if candidate_idx != -1]
    
    def filter_top(self, accumulator) -> None:
        """Apply accumulator to the best candidate for each task.
        
        Iterates over all tasks and calls accumulator.append(task_id, candidate) for each best candidate.
        """
        for task_id in self.get_all_task_ids():
            best_candidate = self.get_best_candidate_for_task(task_id)
            if best_candidate is not None:
                accumulator.append(task_id, best_candidate)
    
    def filter_by_task(self, selector) -> List['Candidate']:
        """Delegate filtering to selector with task score data from ScoreMatrix.
        
        ScoreMatrix provides the best candidates it has stored for each task.
        Selector operates only on the data provided, never accessing pool directly.
        
        Args:
            selector: Selection strategy that implements filter(task_score_data)
            
        Returns:
            List of candidates selected by the selector
        """
        # Prepare task score data using ScoreMatrix's stored best candidates
        task_score_data = {}
        
        for task_id in self.get_all_task_ids():
            best_candidate = self.get_best_candidate_for_task(task_id)
            if best_candidate is not None:
                score = best_candidate.task_score(task_id)
                # ScoreMatrix provides best candidate for each task
                task_score_data[task_id] = [(best_candidate, score)]
        
        # Delegate to selector with prepared data
        return selector.filter(task_score_data)
    
    def filter_by_candidate(self, selector) -> List['Candidate']:
        """Delegate filtering to selector with array of best-performing candidates.
        
        ScoreMatrix provides the list of candidates that perform best on at least one task.
        Selector operates only on the candidates array provided.
        
        Args:
            selector: Selection strategy that implements filter(candidates_array)
            
        Returns:
            List of candidates selected by the selector
        """
        # Provide the candidates that perform best on at least one task
        return selector.filter(self.candidates)
    
    def _add_or_find_candidate(self, candidate: 'Candidate') -> int:
        """Add candidate to candidates list if not present, return its index."""
        try:
            return self.candidates.index(candidate)
        except ValueError:
            # Candidate not in list, add it
            self.candidates.append(candidate)
            return len(self.candidates) - 1
    
