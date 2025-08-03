"""Enhanced CandidatePool for the GEPA strategy framework.

Provides generation-aware candidate management with integrated ScoreMatrix
for fast access to best candidates per task.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Callable
import weakref

from .candidate import Candidate
from .score_matrix import ScoreMatrix
from .cohort import Cohort


class CandidatePool:
    """Framework-aware candidate pool with integrated task-based scoring.
    
    Manages candidates across generations with integrated ScoreMatrix
    for fast access to best candidates per task.
    """
    
    def __init__(self):
        # Primary storage - candidates directly
        self.candidates: List[Candidate] = []
        
        # Generation indexing for efficient access
        self.candidates_by_generation: Dict[int, List[Candidate]] = defaultdict(list)
        
        # Integrated task-based scoring matrix
        self.score_matrix = ScoreMatrix()
    
    def promote(self, cohort: Cohort) -> None:
        """Promote a cohort to the pool and update score matrix.
        
        Candidates should already have their task_scores populated from evaluation.
        """
        for candidate in cohort.candidates:
            self.candidates.append(candidate)
            self.candidates_by_generation[cohort.iteration_id].append(candidate)
        
        # Update score matrix with the entire cohort at once
        self.score_matrix.update_scores(cohort)
    
    def append(self, candidate: Candidate) -> ScoreMatrix:
        """Add a candidate to the pool.
        
        The candidate should already have its task_scores populated from evaluation.
        
        Returns:
            The updated score matrix
        """
        # Add candidate to pool
        self.candidates.append(candidate)
        self.candidates_by_generation[candidate.generation_number].append(candidate)
        
        # Update score matrix with this new candidate
        from .cohort import Cohort
        single_candidate_cohort = Cohort(candidate)
        self.score_matrix.update_scores(single_candidate_cohort)
        
        return self.score_matrix
    
    def append_with_scores(self, candidate: Candidate, task_scores: Dict[int, float]) -> ScoreMatrix:
        """Add a candidate to the pool with explicit task scores (for backward compatibility).
        
        This method sets the scores in the candidate first, then adds it.
        """
        # Store scores in the candidate  
        candidate.set_task_scores(task_scores)
        return self.append(candidate)
    
    
    
    
    
    def filter_by_task_scores(self, selector) -> List[Candidate]:
        """Filter candidates based on their task-by-task score performance.
        
        Delegates to score_matrix because the matrix is updated each time
        a candidate is added to the pool, maintaining score consistency.
        
        Args:
            selector: Selection strategy that receives task score data
            
        Returns:
            List of candidates selected based on task performance
        """
        return self.score_matrix.filter_by_task(selector)
    
    def filter_best_scores(self, selector) -> List[Candidate]:
        """Filter from candidates with the best scores (those who excel in at least one task).
        
        Delegates to score_matrix because the matrix is updated each time
        a candidate is added to the pool, maintaining the best performers list.
        
        Args:
            selector: Selection strategy that receives array of best candidates
            
        Returns:
            List of candidates selected from the high performers
        """
        return self.score_matrix.filter_by_candidate(selector)
    
    def filter(self, selector) -> List[Candidate]:
        """Filter all candidates in the pool.
        
        Direct filtering of all candidates without score considerations.
        
        Args:
            selector: Selection strategy that receives array of all candidates
            
        Returns:
            List of candidates selected from the entire pool
        """
        return selector.filter(self.candidates)
    
    def select_one(self, selector) -> Candidate:
        """Select single candidate using selector."""
        if hasattr(selector, 'select_one_stochastic'):
            return selector.select_one_stochastic(self)
        else:
            candidates = selector.select_from_pool(self)
            return candidates[0] if candidates else None
    
    def filter_top(self, accumulator) -> None:
        """Apply accumulator to the best candidate per task.
        
        Delegates to score_matrix which iterates over top candidates per task.
        """
        self.score_matrix.filter_top(accumulator)
    
    def size(self) -> int:
        """Total number of candidates in pool."""
        return len(self.candidates)
    
    def generation_count(self) -> int:
        """Number of generations in pool."""
        return len(self.candidates_by_generation)
    
    def select_best(self, metric, evaluation_data: List, n: int = 1) -> List[Candidate]:
        """Select the best N candidates from the entire pool."""
        if not self.candidates:
            return []
        
        # Evaluate all candidates and sort by performance
        candidate_scores = []
        for candidate in self.candidates:
            score = candidate.evaluate_on_batch(evaluation_data, metric)
            candidate_scores.append((candidate, score))
        
        # Sort by score (descending) and return top N
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in candidate_scores[:n]]
    
    
