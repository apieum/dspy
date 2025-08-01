"""Candidate pool and lineage management for GEPA optimization."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import dspy
from dspy import Module

from .structures import ScoreMatrix, FeedbackResult


@dataclass
class CandidateLineage:
    """Track ancestry and evolutionary history of candidates."""
    candidate_id: int
    parent_id: Optional[int] = None
    generation: int = 0
    mutation_type: str = "initial"
    creation_iteration: int = 0
    fitness_history: List[float] = None

    def __post_init__(self):
        if self.fitness_history is None:
            self.fitness_history = []


class CandidatePool:
    """Encapsulates candidates, scores, and lineages in a single object.
    
    Manages the state of all candidates in the GEPA optimization process,
    providing clean access to candidates, their performance scores, and
    evolutionary lineages. Includes extension points for future per-candidate
    feedback and over-specialization detection.
    """
    
    def __init__(self):
        self.candidates: List[Module] = []
        self.scores: ScoreMatrix = ScoreMatrix()
        self.lineages: Dict[int, CandidateLineage] = {}
        # Extension point for future candidate-specific feedback
        self._candidate_feedback: Dict[int, FeedbackResult] = {}
        self._next_candidate_id = 0
    
    def add_candidate(self, candidate: Module, lineage: Optional[CandidateLineage] = None) -> int:
        """Add a new candidate to the pool.
        
        Args:
            candidate: The candidate module to add
            lineage: Optional lineage information for tracking evolution
            
        Returns:
            The candidate ID assigned to this candidate
        """
        candidate_id = len(self.candidates)
        self.candidates.append(candidate)
        
        if lineage is not None:
            self.lineages[candidate_id] = lineage
        
        return candidate_id
    
    def get_candidates(self) -> List[Module]:
        """Get all candidates in the pool."""
        return self.candidates.copy()
    
    def get_candidate(self, candidate_id: int) -> Optional[Module]:
        """Get a specific candidate by ID."""
        if 0 <= candidate_id < len(self.candidates):
            return self.candidates[candidate_id]
        return None
    
    def get_scores(self) -> ScoreMatrix:
        """Get the score matrix for all candidates."""
        return self.scores
    
    def get_lineages(self) -> Dict[int, CandidateLineage]:
        """Get lineage information for all candidates."""
        return self.lineages.copy()
    
    def get_lineage(self, candidate_id: int) -> Optional[CandidateLineage]:
        """Get lineage information for a specific candidate."""
        return self.lineages.get(candidate_id)
    
    def size(self) -> int:
        """Number of candidates in pool."""
        return len(self.candidates)
    
    def set_candidate_feedback(self, candidate_id: int, feedback: FeedbackResult):
        """Set feedback for a specific candidate (extension point)."""
        self._candidate_feedback[candidate_id] = feedback
    
    def get_candidate_feedback(self, candidate_id: int) -> Optional[FeedbackResult]:
        """Get feedback for a specific candidate (extension point)."""
        return self._candidate_feedback.get(candidate_id)
    
    def clear(self):
        """Clear all candidates, scores, and lineages."""
        self.candidates.clear()
        self.scores = ScoreMatrix()
        self.lineages.clear()
        self._candidate_feedback.clear()
        self._next_candidate_id = 0
    
    def __len__(self) -> int:
        """Support len() operation."""
        return len(self.candidates)
    
    def __iter__(self):
        """Support iteration over candidates."""
        return iter(self.candidates)
    
    # Score matrix delegation methods
    def set_score(self, candidate_id: int, task_idx: int, score: float):
        """Set score for a candidate on a specific task."""
        self.scores.set_score(candidate_id, task_idx, score)
    
    def get_score(self, candidate_id: int, task_idx: int) -> Optional[float]:
        """Get score for a candidate on a specific task."""
        return self.scores.get_score(candidate_id, task_idx)
    
    def get_candidate_scores(self, candidate_id: int) -> Dict[int, float]:
        """Get all scores for a specific candidate."""
        return self.scores.get_candidate_scores(candidate_id)
    
    def compute_average_score(self, candidate_id: int) -> float:
        """Compute average score for a candidate across all tasks."""
        return self.scores.compute_average_score(candidate_id)
    
    def remove_candidate(self, candidate_id: int):
        """Remove a candidate from the pool."""
        if 0 <= candidate_id < len(self.candidates):
            # Remove from candidates list
            self.candidates.pop(candidate_id)
            # Remove from lineages
            if candidate_id in self.lineages:
                del self.lineages[candidate_id]
            # Remove from candidate feedback
            if candidate_id in self._candidate_feedback:
                del self._candidate_feedback[candidate_id]
            # Note: ScoreMatrix scores are indexed by candidate_id, 
            # so removing from the middle will cause index mismatch.
            # This is a limitation of the current design that would need 
            # more sophisticated index management to fix properly.
    
    def evaluate(self, evaluation_function, tasks, skip_evaluated=True):
        """Evaluate candidates using visitor pattern.
        
        Args:
            evaluation_function: Function that takes (candidate, task_idx, task) and returns score
            tasks: List of tasks/examples to evaluate on
            skip_evaluated: Skip already evaluated candidate-task pairs
            
        Returns:
            Number of evaluations performed
        """
        evaluations_performed = 0
        
        for candidate_id, candidate in enumerate(self.candidates):
            if skip_evaluated:
                # Skip if already fully evaluated
                candidate_scores = self.get_candidate_scores(candidate_id)
                if len(candidate_scores) >= len(tasks):
                    continue
            
            for task_idx, task in enumerate(tasks):
                if skip_evaluated:
                    # Skip if this specific candidate-task pair already evaluated
                    if self.get_score(candidate_id, task_idx) is not None:
                        continue
                
                try:
                    # Use visitor pattern - delegate evaluation to provided function
                    score = evaluation_function(candidate, task_idx, task)
                    self.set_score(candidate_id, task_idx, score)
                    evaluations_performed += 1
                    
                except Exception as e:
                    # Handle evaluation failures gracefully
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to evaluate candidate {candidate_id} on task {task_idx}: {e}")
                    self.set_score(candidate_id, task_idx, 0.0)
        
        return evaluations_performed
    
    def evaluate_specific_candidates(self, evaluation_function, tasks, candidate_ids, skip_evaluated=True):
        """Evaluate specific candidates using visitor pattern.
        
        Args:
            evaluation_function: Function that takes (candidate, task_idx, task) and returns score
            tasks: List of tasks/examples to evaluate on
            candidate_ids: List of specific candidate IDs to evaluate
            skip_evaluated: Skip already evaluated candidate-task pairs
            
        Returns:
            Number of evaluations performed
        """
        evaluations_performed = 0
        
        for candidate_id in candidate_ids:
            if candidate_id >= len(self.candidates):
                continue
                
            candidate = self.candidates[candidate_id]
            
            if skip_evaluated:
                # Skip if already fully evaluated
                candidate_scores = self.get_candidate_scores(candidate_id)
                if len(candidate_scores) >= len(tasks):
                    continue
            
            for task_idx, task in enumerate(tasks):
                if skip_evaluated:
                    # Skip if this specific candidate-task pair already evaluated
                    if self.get_score(candidate_id, task_idx) is not None:
                        continue
                
                try:
                    # Use visitor pattern - delegate evaluation to provided function
                    score = evaluation_function(candidate, task_idx, task)
                    self.set_score(candidate_id, task_idx, score)
                    evaluations_performed += 1
                    
                except Exception as e:
                    # Handle evaluation failures gracefully
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to evaluate candidate {candidate_id} on task {task_idx}: {e}")
                    self.set_score(candidate_id, task_idx, 0.0)
        
        return evaluations_performed