"""Promotion evaluation implementation."""

from typing import List, Callable
import dspy
from .evaluator import Evaluator
from ..data.cohort import Cohort, FilteredCohort


class PromotionEvaluator(Evaluator):
    """Evaluator that promotes promising candidates."""
    
    def __init__(self, metric: Callable, promotion_threshold: float = 0.5):
        self.metric = metric
        self.promotion_threshold = promotion_threshold
        
    def evaluate(self, cohort: Cohort, 
                evaluation_data: List[dspy.Example]) -> FilteredCohort:
        
        promoted_candidates = []
        
        for candidate in cohort.candidates:
            # Evaluate candidate performance
            score = candidate.evaluate_on_batch(evaluation_data, self.metric)
            
            # Promote if above threshold
            if score >= self.promotion_threshold:
                promoted_candidates.append(candidate)
                
        return FilteredCohort(
            candidates=promoted_candidates,
            filtered_count=len(cohort.candidates) - len(promoted_candidates)
        )
        
    def get_metric(self) -> Callable:
        return self.metric