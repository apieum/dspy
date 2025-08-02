"""Diversity selection implementation."""

from typing import List
from .selection import Selection
from ..data.candidate import Candidate
from ..data.score_matrix import ScoreMatrix
from ..data.candidate_pool import CandidatePool


class DiversitySelection(Selection):
    """Selection that maintains population diversity."""
    
    def __init__(self, diversity_weight: float = 0.3, keep_top_n: int = 5):
        self.diversity_weight = diversity_weight
        self.keep_top_n = keep_top_n
        
    def filter(self, candidate_pool: CandidatePool, score_matrix: ScoreMatrix) -> List[Candidate]:
        # Use filtering for diversity-aware filtering
        from ..filtering import ParetoFrontier, BalancedTop
        
        # First get Pareto frontier candidates
        pareto_filtering = ParetoFrontier()
        pareto_candidates = candidate_pool.filter_best(pareto_filtering)
        
        # If we need more candidates, supplement with balanced top performers
        if len(pareto_candidates) < self.keep_top_n:
            balanced_filtering = BalancedTop(keep_top_n=self.keep_top_n)
            balanced_candidates = candidate_pool.filter_top(candidate_pool.size(), balanced_filtering)
            
            # Add candidates not already in Pareto set
            pareto_ids = {c.candidate_id for c in pareto_candidates}
            for candidate in balanced_candidates:
                if candidate.candidate_id not in pareto_ids and len(pareto_candidates) < self.keep_top_n:
                    pareto_candidates.append(candidate)
        
        return pareto_candidates[:self.keep_top_n]