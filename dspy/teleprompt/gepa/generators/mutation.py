"""Mutation-based candidate generation."""

import logging
from typing import Iterable, List, Optional

import dspy
from dspy.teleprompt.utils import get_signature, set_signature

from ..data.candidate_pool import CandidatePool
from ..data.structures import FeedbackResult
from .base import CandidateGenerator

logger = logging.getLogger(__name__)


class MutationGenerator(CandidateGenerator):
    """Paper-compliant mutation generator using reflective prompt mutation.
    
    Implements the mutation strategy from GEPA paper, selecting a parent
    candidate and improving it through reflective prompt mutation based
    on feedback from generic feedback data.
    """
    
    def __init__(self, prompt_mutator: "PromptMutator", 
                 module_selector: "ModuleSelector",
                 feedback_collector: "FeedbackCollector"):
        """Initialize mutation generator with required components.
        
        Args:
            prompt_mutator: Strategy for mutating prompts based on feedback
            module_selector: Strategy for selecting which module to mutate
            feedback_collector: Collector for gathering performance feedback
        """
        self.prompt_mutator = prompt_mutator
        self.module_selector = module_selector
        self.feedback_collector = feedback_collector
    
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[dspy.Example], 
                          iteration: int) -> List[dspy.Module]:
        """Generate new candidate through mutation of selected parent.
        
        Follows paper's approach:
        1. Select parent candidate based on performance
        2. Collect feedback on parent using generic feedback_data
        3. Apply reflective prompt mutation
        4. Return improved candidate
        """
        if candidate_pool.size() == 0:
            return []
        
        try:
            # Step 1: Select parent candidate based on performance
            parent_idx = self._select_parent_candidate(candidate_pool)
            parent_candidate = candidate_pool.get_candidate(parent_idx)
            
            if parent_candidate is None:
                return []
            
            # Step 2: Collect feedback on parent's performance
            feedback_data_list = list(feedback_data)
            if not feedback_data_list:
                return []
            
            feedback_result = self.feedback_collector.collect_feedback(
                parent_candidate, feedback_data_list, lambda ex, pred, trace=None: 1.0  # Placeholder metric
            )
            
            # Step 3: Apply reflective prompt mutation
            mutated_candidate = self._mutate_candidate(parent_candidate, feedback_result)
            
            if mutated_candidate is None:
                return []
            
            return [mutated_candidate]
            
        except Exception as e:
            logger.warning(f"Mutation generation failed: {e}")
            return []
    
    def _select_parent_candidate(self, candidate_pool: CandidatePool) -> int:
        """Select parent candidate for mutation based on performance."""
        scores = candidate_pool.get_scores()
        candidates = candidate_pool.get_candidates()
        
        # Find candidate with best average score
        best_idx = 0
        best_score = -float('inf')
        
        for i in range(len(candidates)):
            avg_score = scores.compute_average_score(i)
            if avg_score > best_score:
                best_score = avg_score
                best_idx = i
        
        return best_idx
    
    def _mutate_candidate(self, parent_candidate: dspy.Module, feedback_result: FeedbackResult) -> Optional[dspy.Module]:
        """Apply mutation to parent candidate using feedback."""
        try:
            # Create a copy of the parent
            mutated_candidate = parent_candidate.deepcopy()
            
            # Select module to mutate
            module_idx = self.module_selector.select_module(mutated_candidate)
            predictors = mutated_candidate.predictors()
            
            if module_idx < len(predictors):
                predictor = predictors[module_idx]
                current_signature = get_signature(predictor)
                
                # Apply reflective mutation
                improved_signature = self.prompt_mutator.mutate_signature(current_signature, feedback_result)
                set_signature(predictor, improved_signature)
            
            return mutated_candidate
            
        except Exception as e:
            logger.warning(f"Candidate mutation failed: {e}")
            return None