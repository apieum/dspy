"""Mutation-based candidate generation with reflective prompt mutation."""

import logging
from typing import List, Optional
import dspy
from dspy.teleprompt.utils import get_signature, set_signature

from .generator import Generator
from .reflective_mutation import ReflectiveMutation, PromptMutator
from ..data.candidate import Candidate
from ..evaluation.trace_collector import EnhancedTraceCollector
from ..evaluation.feedback import FeedbackResult
from ..data.cohort import Cohort

logger = logging.getLogger(__name__)


class ModuleSelector:
    """Simple module selector for mutation."""
    
    def select_module(self, program: dspy.Module) -> int:
        """Select module index to mutate. For now, select first predictor."""
        predictors = program.predictors()
        return 0 if predictors else 0


class MutationGenerator(Generator):
    """Paper-compliant mutation generator using reflective prompt mutation.
    
    Implements the mutation strategy from GEPA paper, selecting a parent
    candidate and improving it through reflective prompt mutation based
    on feedback from generic feedback data.
    """
    
    def __init__(self, 
                 prompt_mutator: Optional[PromptMutator] = None,
                 module_selector: Optional[ModuleSelector] = None,
                 feedback_collector: Optional[EnhancedTraceCollector] = None,
                 mutation_rate: float = 0.3, 
                 population_size: int = 10):
        """Initialize mutation generator with required components.
        
        Args:
            prompt_mutator: Strategy for mutating prompts based on feedback
            module_selector: Strategy for selecting which module to mutate
            feedback_collector: Collector for gathering performance feedback
            mutation_rate: Probability of applying mutation
            population_size: Number of candidates to generate
        """
        self.prompt_mutator = prompt_mutator or ReflectiveMutation()
        self.module_selector = module_selector or ModuleSelector()
        self.feedback_collector = feedback_collector or EnhancedTraceCollector()
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        # This will be set during prepare_for_compilation()
        self.feedback_data: List[dspy.Example] = []
        
    def generate(self, parent_candidates: List[Candidate], iteration: int, budget=None) -> Cohort:
        """Generate new candidates through mutation of selected parents.
        
        Follows paper's approach:
        1. Select parent candidate based on performance
        2. Collect feedback on parent using generic feedback_data
        3. Apply reflective prompt mutation
        4. Return improved candidates
        """
        new_candidates = []
        
        if not parent_candidates or not self.feedback_data:
            return Cohort(*new_candidates)
        
        for i in range(self.population_size):
            try:
                # Step 1: Select parent candidate 
                parent_candidate = self._select_parent_candidate(parent_candidates)
                
                if parent_candidate is None:
                    continue
                
                # Step 2: Collect feedback on parent's performance
                feedback_result = self.feedback_collector.collect_feedback(
                    parent_candidate.module, self.feedback_data, self._simple_metric
                )
                
                # Step 3: Apply reflective prompt mutation
                mutated_candidate = self._mutate_candidate(parent_candidate, feedback_result, iteration, budget)
                
                if mutated_candidate is not None:
                    new_candidates.append(mutated_candidate)
                    
            except Exception as e:
                logger.warning(f"Mutation failed for candidate {i}: {e}")
                continue
        
        return Cohort(*new_candidates)

    def _select_parent_candidate(self, parent_candidates: List[Candidate]) -> Optional[Candidate]:
        """Select parent candidate based on performance."""
        if not parent_candidates:
            return None
        
        # Select candidate with highest average score
        best_candidate = None
        best_score = -1.0
        
        for candidate in parent_candidates:
            avg_score = candidate.average_task_score()
            if avg_score > best_score:
                best_score = avg_score
                best_candidate = candidate
        
        return best_candidate or parent_candidates[0]

    def _mutate_candidate(self, parent_candidate: Candidate, feedback_result: FeedbackResult, iteration: int, budget=None) -> Optional[Candidate]:
        """Mutate candidate using reflective prompt mutation."""
        try:
            # Get predictors from parent module
            parent_module = parent_candidate.module
            predictors = parent_module.predictors()
            
            if not predictors:
                logger.warning("No predictors found in parent module")
                return None
            
            # Select module to mutate
            module_idx = self.module_selector.select_module(parent_module)
            if module_idx >= len(predictors):
                module_idx = 0
            
            predictor = predictors[module_idx]
            
            # Get current signature
            current_signature = get_signature(predictor)
            
            # Apply reflective mutation (LLM call happens here)
            new_signature = self.prompt_mutator.mutate_signature(current_signature, feedback_result)
            
            # Track budget for generation
            if budget is not None:
                budget.spend_on_generation(parent_candidate.module, {"type": "mutation", "iteration": iteration})
            
            # Create mutated module
            mutated_module = parent_module.deepcopy()
            mutated_predictors = mutated_module.predictors()
            
            if module_idx < len(mutated_predictors):
                set_signature(mutated_predictors[module_idx], new_signature)
            
            # Mark parent as having produced a child
            parent_candidate.had_child = True
            
            # Create new candidate
            new_candidate = Candidate(
                module=mutated_module,
                parents=[parent_candidate],
                generation_number=iteration
            )
            
            return new_candidate
            
        except Exception as e:
            logger.warning(f"Failed to mutate candidate: {e}")
            return None

    def _simple_metric(self, example: dspy.Example, prediction, trace=None) -> float:
        """Simple metric for feedback collection."""
        try:
            # Check if prediction has expected output field
            if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
                expected = str(example.answer).lower().strip()
                actual = str(prediction.answer).lower().strip()
                return 1.0 if expected == actual else 0.0
            
            # Fallback: check any string field
            for field_name in ['answer', 'output', 'result', 'response']:
                if hasattr(example, field_name) and hasattr(prediction, field_name):
                    expected = str(getattr(example, field_name)).lower().strip()
                    actual = str(getattr(prediction, field_name)).lower().strip()
                    return 1.0 if expected == actual else 0.0
            
            return 0.0
        except Exception:
            return 0.0
    
    def generate_from_parents(self, parent_candidates: List[Candidate]) -> Cohort:
        """Generate new candidates from parent candidates (simplified interface).
        
        Used by ParetoFrontier.generate() method for dependency injection pattern.
        """
        return self.generate(parent_candidates, iteration=0)
    
    def create_empty_cohort(self) -> Cohort:
        """Create an empty cohort of the type this generator produces."""
        return Cohort()
    
    def start_compilation(self, student: dspy.Module, training_data: List[dspy.Example]) -> None:
        """Prepare generator with training dataset when compilation begins.
        
        Uses training data as feedback data for reflective mutation.
        """
        self.feedback_data = training_data