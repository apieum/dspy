"""Elitist selection implementation."""

from typing import List
from .selection import Selection
from ..data.candidate import Candidate
from ..data.score_matrix import ScoreMatrix
from ..data.candidate_pool import CandidatePool


class ElitistSelection(Selection):
    """Selection that keeps top N performers."""
    
    def __init__(self, keep_top_n: int = 5):
        self.keep_top_n = keep_top_n
        
    def filter(self, candidate_pool: CandidatePool, score_matrix: ScoreMatrix) -> List[Candidate]:
        # Use filtering to get top performing candidates by balanced scores
        from ..filtering import BalancedTop
        
        filtering = BalancedTop(keep_top_n=self.keep_top_n)
        return candidate_pool.filter_top(candidate_pool.size(), filtering)