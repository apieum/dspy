"""Diversity filtering implementation."""

from typing import Dict, List, Tuple
from .filtering import Filtering


class Diversity(Filtering):
    """Filtering that selects diverse candidates across different tasks."""
    
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