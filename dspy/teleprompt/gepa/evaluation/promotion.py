"""Promotion evaluation implementation."""

from typing import List, Callable
import dspy
from .evaluator import Evaluator
from ..generation.generation import Generation


class PromotionEvaluator(Evaluator):
    """Evaluator that promotes promising candidates."""
    
    def __init__(self, metric: Callable, promotion_threshold: float = 0.5):
        self.metric = metric
        self.promotion_threshold = promotion_threshold
        
    def evaluate(self, generation: Generation, 
                evaluation_data: List[dspy.Example]) -> Generation:
        
        promoted_candidates = []
        
        for candidate in generation.candidates:
            # Evaluate candidate performance
            score = candidate.evaluate_on_batch(evaluation_data, self.metric)
            
            # Promote if above threshold
            if score >= self.promotion_threshold:
                promoted_candidates.append(candidate)
                
        return Generation(
            candidates=promoted_candidates,
            generation_id=generation.generation_id,
            iteration=generation.iteration
        )
        
    def get_metric(self) -> Callable:
        return self.metric