"""Pareto scoring implementation."""

from typing import List, Callable
import dspy
from .scoring import Scoring
from ..data.candidate import Candidate
from ..data.score_matrix import ScoreMatrix
from ..data.candidate_pool import CandidatePool


class ParetoScoring(Scoring):
    """Scoring using Pareto-frontier evaluation."""
    
    def __init__(self, metric: Callable):
        self.metric = metric
        
    def calculate_scores(self, candidates: List[Candidate], 
                        data: List[dspy.Example], 
                        candidate_pool: CandidatePool) -> ScoreMatrix:
        """Evaluate candidates on tasks and add them to the candidate pool."""
        
        for candidate in candidates:
            # Evaluation phase: score candidate on each task (minibatch)
            total_score = 0.0
            
            for task_id, example in enumerate(data):
                score = candidate.evaluate_on_example(example, self.metric)
                # Store score in candidate during evaluation
                candidate.set_task_score(task_id, score)
                total_score += score
                
            # Store average as overall fitness
            if data:
                avg_score = total_score / len(data)
                candidate.add_fitness_score(avg_score)
            
            # Add candidate to pool - pool reads candidate's scores and updates matrix
            candidate_pool.append(candidate)
        
        # Return the candidate pool's score matrix
        return candidate_pool.score_matrix
        
    def get_metric(self) -> Callable:
        return self.metric