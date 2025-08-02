"""Top scores filtering implementation."""

from typing import Dict, List, Tuple
from .filtering import Filtering


class TopScores(Filtering):
    """Filtering that selects candidates with highest scores for specific tasks."""
    
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