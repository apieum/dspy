"""System-Aware Merge crossover implementation from GEPA paper Appendix F."""

import random
import logging
from typing import List, Optional, Dict, Tuple
import dspy
from dspy.teleprompt.utils import get_signature, set_signature
from .generator import Generator
from ..data.candidate import Candidate
from ..data.cohort import Cohort
from ..evaluation.trace_collector import EnhancedTraceCollector

logger = logging.getLogger(__name__)


class SystemAwareMergeGenerator(Generator):
    """System-Aware Merge crossover generator implementing Algorithm 4 from GEPA paper.
    
    Implements sophisticated module-wise crossover that:
    1. Checks for complementary evolution between parents using DESIRABLE function
    2. Uses ancestry-aware selection to avoid inbreeding
    3. Performs intelligent module-wise combination based on performance
    """
    
    def __init__(self, 
                 crossover_rate: float = 0.7, 
                 population_size: int = 10,
                 feedback_collector: Optional[EnhancedTraceCollector] = None):
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.feedback_collector = feedback_collector or EnhancedTraceCollector()
        # Training data will be set during compilation
        self.feedback_data: List[dspy.Example] = []
        
    def generate(self, parent_candidates: List[Candidate], iteration: int, budget=None) -> Cohort:
        """Generate new candidates using System-Aware Merge crossover."""
        new_candidates = []
        
        if len(parent_candidates) < 2 or not self.feedback_data:
            return Cohort(*new_candidates)
        
        for i in range(self.population_size):
            if random.random() < self.crossover_rate:
                try:
                    # Select two parents using ancestry-aware selection
                    parent1, parent2 = self._select_crossover_parents(parent_candidates)
                    
                    if parent1 is None or parent2 is None:
                        continue
                    
                    # Check if parents have complementary evolution using DESIRABLE
                    if not self._desirable(parent1, parent2):
                        continue
                    
                    # Perform System-Aware Merge
                    child_candidate = self._system_aware_merge(parent1, parent2, iteration, budget)
                    
                    if child_candidate is not None:
                        # Mark both parents as having produced a child
                        parent1.had_child = True
                        parent2.had_child = True
                        new_candidates.append(child_candidate)
                        
                except Exception as e:
                    logger.warning(f"Crossover failed for candidate {i}: {e}")
                    continue
                    
        return Cohort(*new_candidates)
    
    def _select_crossover_parents(self, parent_candidates: List[Candidate]) -> Tuple[Optional[Candidate], Optional[Candidate]]:
        """Select two parents for crossover using ancestry-aware selection.
        
        Avoids selecting parents that are too closely related (same lineage).
        """
        if len(parent_candidates) < 2:
            return None, None
        
        # Try to find unrelated parents
        for _ in range(10):  # Max attempts to avoid infinite loop
            parent1 = random.choice(parent_candidates)
            parent2 = random.choice(parent_candidates)
            
            if parent1 != parent2 and not self._are_closely_related(parent1, parent2):
                return parent1, parent2
        
        # Fallback: select any two different parents
        if len(parent_candidates) >= 2:
            candidates = random.sample(parent_candidates, 2)
            return candidates[0], candidates[1]
        
        return None, None
    
    def _are_closely_related(self, candidate1: Candidate, candidate2: Candidate) -> bool:
        """Check if two candidates are closely related (share recent ancestry)."""
        # Check if one is ancestor of the other
        if self._is_ancestor(candidate1, candidate2) or self._is_ancestor(candidate2, candidate1):
            return True
        
        # Check if they share a common parent
        parents1 = set(candidate1.parents) if candidate1.parents else set()
        parents2 = set(candidate2.parents) if candidate2.parents else set()
        
        return len(parents1.intersection(parents2)) > 0
    
    def _is_ancestor(self, potential_ancestor: Candidate, descendant: Candidate) -> bool:
        """Check if potential_ancestor is an ancestor of descendant."""
        if not descendant.parents:
            return False
        
        if potential_ancestor in descendant.parents:
            return True
        
        # Recursively check ancestors
        for parent in descendant.parents:
            if self._is_ancestor(potential_ancestor, parent):
                return True
        
        return False
    
    def _desirable(self, parent1: Candidate, parent2: Candidate) -> bool:
        """DESIRABLE function (Algorithm 3) - check for complementary evolution.
        
        Two candidates have complementary evolution if they excel at different tasks,
        making their combination potentially beneficial.
        """
        try:
            # Get task performance for both parents
            scores1 = parent1.task_scores
            scores2 = parent2.task_scores
            
            if not scores1 or not scores2:
                return True  # Allow crossover if we don't have enough data
            
            # Convert to common format (list or dict)
            if isinstance(scores1, dict) and isinstance(scores2, dict):
                # Find common tasks
                common_tasks = set(scores1.keys()).intersection(set(scores2.keys()))
                if not common_tasks:
                    return True
                
                # Check if parents excel at different tasks
                complementary_count = 0
                total_tasks = len(common_tasks)
                
                for task_id in common_tasks:
                    score1 = scores1[task_id]
                    score2 = scores2[task_id]
                    
                    # If one significantly outperforms the other, they're complementary
                    if abs(score1 - score2) > 0.1:  # Threshold for "significant" difference
                        complementary_count += 1
                
                # Parents are desirable if they show complementary performance
                complementary_ratio = complementary_count / total_tasks if total_tasks > 0 else 0
                return complementary_ratio > 0.3  # At least 30% of tasks show complementary performance
                
            elif isinstance(scores1, list) and isinstance(scores2, list):
                min_len = min(len(scores1), len(scores2))
                if min_len == 0:
                    return True
                
                complementary_count = 0
                for i in range(min_len):
                    if abs(scores1[i] - scores2[i]) > 0.1:
                        complementary_count += 1
                
                complementary_ratio = complementary_count / min_len
                return complementary_ratio > 0.3
            
            return True  # Default to allowing crossover
            
        except Exception as e:
            logger.warning(f"DESIRABLE function failed: {e}")
            return True  # Default to allowing crossover
    
    def _system_aware_merge(self, parent1: Candidate, parent2: Candidate, iteration: int, budget=None) -> Optional[Candidate]:
        """Perform System-Aware Merge (Algorithm 4) - module-wise crossover."""
        try:
            # Get predictors from both parents
            predictors1 = parent1.module.predictors()
            predictors2 = parent2.module.predictors()
            
            if not predictors1 and not predictors2:
                return None
            
            # Create child module by copying parent1 as base
            child_module = parent1.module.deepcopy()
            child_predictors = child_module.predictors()
            
            # For each module position, select best performing parent's module
            max_modules = max(len(predictors1), len(predictors2))
            
            for module_idx in range(max_modules):
                try:
                    # Get module performance for both parents on this module
                    perf1 = self._evaluate_module_performance(parent1, module_idx)
                    perf2 = self._evaluate_module_performance(parent2, module_idx)
                    
                    # Select better performing parent's module
                    if perf2 > perf1 and module_idx < len(predictors2) and module_idx < len(child_predictors):
                        # Copy module from parent2
                        source_signature = get_signature(predictors2[module_idx])
                        set_signature(child_predictors[module_idx], source_signature)
                        
                except Exception as e:
                    logger.warning(f"Module selection failed for index {module_idx}: {e}")
                    continue
            
            # Track budget for crossover
            if budget is not None:
                budget.spend_on_generation(parent1.module, {"type": "crossover", "iteration": iteration})
            
            # Create new candidate
            child_candidate = Candidate(
                module=child_module,
                parents=[parent1, parent2],
                generation_number=iteration
            )
            
            return child_candidate
            
        except Exception as e:
            logger.warning(f"System-aware merge failed: {e}")
            return None
    
    def _evaluate_module_performance(self, candidate: Candidate, module_idx: int) -> float:
        """Evaluate performance of a specific module within a candidate.
        
        This is a simplified version - the full paper implementation would
        evaluate each module's contribution to overall performance.
        """
        try:
            # Use overall candidate performance as proxy for module performance
            if isinstance(candidate.task_scores, dict):
                scores = list(candidate.task_scores.values())
            elif isinstance(candidate.task_scores, list):
                scores = candidate.task_scores
            else:
                return 0.0
            
            if not scores:
                return 0.0
            
            # Return average score as module performance proxy
            return sum(scores) / len(scores)
            
        except Exception:
            return 0.0
    
    def generate_from_parents(self, parent_candidates: List[Candidate]) -> Cohort:
        """Generate new candidates from parent candidates (simplified interface)."""
        return self.generate(parent_candidates, iteration=0)
    
    def create_empty_cohort(self) -> Cohort:
        """Create an empty cohort of the type this generator produces."""
        return Cohort()
    
    def start_compilation(self, student: dspy.Module, training_data: List[dspy.Example]) -> None:
        """Prepare generator with training dataset when compilation begins."""
        self.feedback_data = training_data


# Backward compatibility alias
CrossoverGenerator = SystemAwareMergeGenerator