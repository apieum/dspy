"""Threshold filtering implementation."""

from typing import List
from .filtering import Filtering


class Threshold(Filtering):
    """Filtering that selects candidates above a performance threshold."""
    
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