"""Pareto frontier filtering implementation."""

from typing import Dict, List, Tuple
from .filtering import Filtering


class ParetoFrontier(Filtering):
    """Filtering that selects candidates forming the Pareto frontier.
    
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