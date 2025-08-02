"""Enhanced CandidatePool for the GEPA strategy framework.

Provides generation-aware candidate management with integrated ScoreMatrix
for fast access to best candidates per task.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Callable
import weakref

from .candidate import Candidate
from .score_matrix import ScoreMatrix
from ..generation.generation import Generation


class CandidatePool:
    """Framework-aware candidate pool with integrated task-based scoring.
    
    Manages candidates across generations with integrated ScoreMatrix
    for fast access to best candidates per task.
    """
    
    def __init__(self):
        # Primary storage
        self.candidates: Dict[int, Candidate] = {}
        self.next_candidate_id = 0
        
        # Generation indexing for efficient access
        self.candidates_by_generation: Dict[int, List[int]] = defaultdict(list)
        self.current_generation_id = 0
        
        # Lineage relationships (for research analysis)
        self.children_by_parent: Dict[int, List[int]] = defaultdict(list)
        self.parent_relationships: Dict[int, List[int]] = {}  # child_id -> parent_ids
        
        # Integrated task-based scoring matrix
        self.score_matrix = ScoreMatrix()
    
    def extend(self, generation: Generation) -> ScoreMatrix:
        """Add a generation to the pool and return score matrix.
        
        Candidates should already have their task_scores populated from evaluation.
        """
        for candidate in generation.candidates:
            self._add_candidate_to_pool(candidate, generation.generation_id)
        
        # Update score matrix with the new candidates
        self.score_matrix.update_scores(generation.candidates)
        
        return self.score_matrix
    
    def add_candidate(self, candidate: Candidate) -> ScoreMatrix:
        """Add a candidate to the pool.
        
        The candidate should already have its task_scores populated from evaluation.
        
        Returns:
            The updated score matrix
        """
        # Add candidate to pool
        generation_id = candidate.generation_number
        candidate_id = self._add_candidate_to_pool(candidate, generation_id)
        
        # Update score matrix with this new candidate
        self.score_matrix.update_scores([candidate])
        
        return self.score_matrix
    
    def add_candidate_with_scores(self, candidate: Candidate, task_scores: Dict[int, float]) -> ScoreMatrix:
        """Add a candidate to the pool with explicit task scores (for backward compatibility).
        
        This method sets the scores in the candidate first, then adds it.
        """
        # Store scores in the candidate  
        candidate.set_task_scores(task_scores)
        return self.add_candidate(candidate)
    
    
    def _add_candidate_to_pool(self, candidate: Candidate, generation_id: int) -> int:
        """Internal method to add candidate with ID assignment and indexing."""
        # Assign candidate ID if not already set
        if candidate.candidate_id is None:
            candidate.candidate_id = self.next_candidate_id
            self.next_candidate_id += 1
        
        candidate_id = candidate.candidate_id
        
        # Store candidate
        self.candidates[candidate_id] = candidate
        
        # Update generation index
        self.candidates_by_generation[generation_id].append(candidate_id)
        
        # Update lineage tracking
        if candidate.parent_ids:
            self.parent_relationships[candidate_id] = candidate.parent_ids.copy()
            for parent_id in candidate.parent_ids:
                self.children_by_parent[parent_id].append(candidate_id)
        
        return candidate_id
    
    def get_candidate(self, candidate_id: int) -> Optional[Candidate]:
        """Get candidate by ID."""
        return self.candidates.get(candidate_id)
    
    def get_all_candidates(self) -> List[Candidate]:
        """Get all candidates in the pool."""
        return list(self.candidates.values())
    
    def get_generation_candidates(self, generation_id: int) -> List[Candidate]:
        """Get all candidates from a specific generation."""
        candidate_ids = self.candidates_by_generation[generation_id]
        return [self.candidates[cid] for cid in candidate_ids if cid in self.candidates]
    
    
    
    def filter_scores(self, strategy) -> List[Candidate]:
        """Apply filtering strategy to task scores.
        
        The pool iterates over each task and passes task_id -> List[scores] 
        to the strategy for filtering.
        """
        task_scores_data = {}
        
        # Collect scores for each task
        for task_id in self.score_matrix.get_all_task_ids():
            candidate_scores = []
            for candidate in self.candidates.values():
                score = candidate.get_task_score(task_id)
                if score is not None:
                    candidate_scores.append((candidate.candidate_id, score))
            task_scores_data[task_id] = candidate_scores
        
        # Pass to strategy for filtering
        selected_candidate_ids = strategy.filter(task_scores_data)
        
        # Return selected candidates
        return [self.candidates[cid] for cid in selected_candidate_ids if cid in self.candidates]
    
    def filter_best(self, strategy) -> List[Candidate]:
        """Apply filtering strategy to best candidates.
        
        The pool iterates over best candidates and passes them to the strategy.
        """
        best_candidates_data = {}
        
        for task_id in self.score_matrix.get_all_task_ids():
            best_candidate = self.score_matrix.get_best_candidate_for_task(task_id)
            if best_candidate is not None:
                score = best_candidate.get_task_score(task_id)
                best_candidates_data[task_id] = (best_candidate.candidate_id, score)
        
        # Pass to strategy for filtering  
        selected_candidate_ids = strategy.filter(best_candidates_data)
        
        # Return selected candidates
        return [self.candidates[cid] for cid in selected_candidate_ids if cid in self.candidates]
    
    def filter_top(self, n: int, strategy) -> List[Candidate]:
        """Apply filtering strategy to top N candidates.
        
        The pool gets top candidates and passes them to the strategy for filtering.
        """
        # Get all candidates with their average scores
        candidate_averages = []
        for candidate in self.candidates.values():
            if candidate.task_scores:
                avg_score = candidate.get_average_task_score()
                candidate_averages.append((candidate.candidate_id, avg_score))
        
        # Sort by average score and take top N
        candidate_averages.sort(key=lambda x: x[1], reverse=True)
        top_candidates_data = candidate_averages[:n]
        
        # Pass to strategy for filtering
        selected_candidate_ids = strategy.filter(top_candidates_data)
        
        # Return selected candidates
        return [self.candidates[cid] for cid in selected_candidate_ids if cid in self.candidates]
    
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
        for candidate in self.candidates.values():
            score = candidate.evaluate_on_batch(evaluation_data, metric)
            candidate_scores.append((candidate, score))
        
        # Sort by score (descending) and return top N
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in candidate_scores[:n]]
    
    
