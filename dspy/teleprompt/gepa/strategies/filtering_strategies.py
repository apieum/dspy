"""Filtering strategies for CandidatePool.

These strategies filter candidates based on different performance criteria
using the pool's filter_* methods.
"""

from typing import Dict, List, Tuple
from abc import ABC, abstractmethod


class FilteringStrategy(ABC):
    """Base class for candidate filtering strategies."""
    
    @abstractmethod
    def filter(self, data) -> List[int]:
        """Filter the provided data and return selected candidate IDs."""
        pass


class ParetoFrontierStrategy(FilteringStrategy):
    """Strategy that selects candidates forming the Pareto frontier.
    
    Selects candidates that are the best performer in at least one task.
    """
    
    def filter(self, best_candidates_data: Dict[int, Tuple[int, float]]) -> List[int]:
        """Filter to get Pareto frontier from best candidates per task.
        
        Args:
            best_candidates_data: task_id -> (candidate_id, score)
            
        Returns:
            List of candidate IDs forming the Pareto frontier
        """
        pareto_candidate_ids = set()
        
        for task_id, (candidate_id, score) in best_candidates_data.items():
            pareto_candidate_ids.add(candidate_id)
        
        return list(pareto_candidate_ids)


class TopScoresStrategy(FilteringStrategy):
    """Strategy that selects candidates with highest scores for specific tasks."""
    
    def __init__(self, target_tasks: List[int], top_n: int = 3):
        self.target_tasks = target_tasks
        self.top_n = top_n
    
    def filter(self, task_scores_data: Dict[int, List[Tuple[int, float]]]) -> List[int]:
        """Filter to get top performers for target tasks.
        
        Args:
            task_scores_data: task_id -> List[(candidate_id, score)]
            
        Returns:
            List of candidate IDs that are top performers for target tasks
        """
        selected_candidates = set()
        
        for task_id in self.target_tasks:
            if task_id in task_scores_data:
                # Sort by score (descending) and take top N
                sorted_candidates = sorted(
                    task_scores_data[task_id], 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for candidate_id, score in sorted_candidates[:self.top_n]:
                    selected_candidates.add(candidate_id)
        
        return list(selected_candidates)


class BalancedTopStrategy(FilteringStrategy):
    """Strategy that selects top N candidates by average performance."""
    
    def __init__(self, keep_top_n: int = 5):
        self.keep_top_n = keep_top_n
    
    def filter(self, top_candidates_data: List[Tuple[int, float]]) -> List[int]:
        """Filter to get top N by average score.
        
        Args:
            top_candidates_data: List[(candidate_id, average_score)] already sorted
            
        Returns:
            List of top N candidate IDs
        """
        return [candidate_id for candidate_id, _ in top_candidates_data[:self.keep_top_n]]


class ThresholdStrategy(FilteringStrategy):
    """Strategy that selects candidates above a performance threshold."""
    
    def __init__(self, min_score: float = 0.7):
        self.min_score = min_score
    
    def filter(self, data) -> List[int]:
        """Filter candidates above threshold.
        
        Works with both task_scores_data and top_candidates_data formats.
        """
        selected_candidates = set()
        
        if isinstance(data, dict):
            # task_scores_data format: task_id -> List[(candidate_id, score)]
            for task_id, candidate_scores in data.items():
                for candidate_id, score in candidate_scores:
                    if score >= self.min_score:
                        selected_candidates.add(candidate_id)
        elif isinstance(data, list):
            # top_candidates_data format: List[(candidate_id, score)]
            for candidate_id, score in data:
                if score >= self.min_score:
                    selected_candidates.add(candidate_id)
        
        return list(selected_candidates)


class DiversityStrategy(FilteringStrategy):
    """Strategy that selects diverse candidates across different tasks."""
    
    def __init__(self, max_per_task: int = 2):
        self.max_per_task = max_per_task
    
    def filter(self, task_scores_data: Dict[int, List[Tuple[int, float]]]) -> List[int]:
        """Filter to get diverse candidates across tasks.
        
        Args:
            task_scores_data: task_id -> List[(candidate_id, score)]
            
        Returns:
            List of candidate IDs selected for diversity
        """
        selected_candidates = set()
        
        for task_id, candidate_scores in task_scores_data.items():
            # Sort by score and take top performers for this task
            sorted_candidates = sorted(
                candidate_scores, 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for candidate_id, score in sorted_candidates[:self.max_per_task]:
                selected_candidates.add(candidate_id)
        
        return list(selected_candidates)