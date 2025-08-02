"""Balanced top filtering implementation."""

from typing import List, Tuple
from .filtering import Filtering


class BalancedTop(Filtering):
    """Filtering that selects top N candidates by average performance."""
    
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