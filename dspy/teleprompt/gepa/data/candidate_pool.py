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
    
    def extend(self, cohort: Cohort) -> ScoreMatrix:
        """Add a cohort to the pool and return score matrix.
        
        Candidates should already have their task_scores populated from evaluation.
        """
        for candidate in cohort.candidates:
            self.candidates.append(candidate)
            self.candidates_by_generation[cohort.iteration_id].append(candidate)
            self.score_matrix.update_scores([candidate])
        
        return self.score_matrix
    
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
        self.score_matrix.update_scores([candidate])
        
        return self.score_matrix
    
    def append_with_scores(self, candidate: Candidate, task_scores: Dict[int, float]) -> ScoreMatrix:
        """Add a candidate to the pool with explicit task scores (for backward compatibility).
        
        This method sets the scores in the candidate first, then adds it.
        """
        # Store scores in the candidate  
        candidate.set_task_scores(task_scores)
        return self.append(candidate)
    
    
    
    
    def filter_scores(self, strategy) -> List[Candidate]:
        """Apply filtering strategy to task scores.
        
        The pool iterates over each task and passes task_id -> List[(candidate, score)] 
        to the strategy for filtering.
        """
        task_scores_data = {}
        
        # Collect scores for each task
        for task_id in self.score_matrix.get_all_task_ids():
            candidate_scores = []
            for candidate in self.candidates:
                score = candidate.task_score(task_id)
                if score is not None:
                    candidate_scores.append((candidate, score))
            task_scores_data[task_id] = candidate_scores
        
        # Pass to strategy for filtering
        selected_candidates = strategy.filter(task_scores_data)
        
        return selected_candidates
    
    def filter(self, selector) -> List[Candidate]:
        """Apply selector for filtering candidates.
        
        Selector encapsulates filtering logic and operates on the pool directly.
        """
        return selector.select_from_pool(self)
    
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
    
    
