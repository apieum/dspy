"""Two-phase evaluation implementation for GEPA Algorithm 1."""

import random
import logging
from typing import List, Callable, Optional, TYPE_CHECKING
import dspy
from .evaluator import Evaluator
from ..data.cohort import NewBorns, Survivors
from ..budget import Budget

if TYPE_CHECKING:
    from ..dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class PromotionEvaluator(Evaluator):
    """
    Implements the two-phase evaluation from GEPA Algorithm 1.
    
    Phase 1: Validate new candidates against their parents on minibatch
    Phase 2: Run full evaluation on Pareto dataset for validated candidates
    """

    def __init__(self, metric: Callable, minibatch_size: int = 3):
        """Initialize two-phase evaluator.
        
        Args:
            metric: Function to evaluate predictions against examples
            minibatch_size: Size of validation minibatch
        """
        self.metric = metric
        self.minibatch_size = minibatch_size
        self.dataset_manager: Optional["DatasetManager"] = None

    def start_compilation(self, student: dspy.Module, dataset_manager: "DatasetManager") -> None:
        """Prepares the evaluator with dataset manager.
        
        Args:
            student: Base module (not used in current implementation)
            dataset_manager: Centralized manager for all dataset operations
        """
        self.dataset_manager = dataset_manager
        pareto_size = len(dataset_manager.get_pareto_set())
        logger.debug(f"PromotionEvaluator prepared with {pareto_size} pareto tasks and dataset manager")

    def evaluate(self, new_borns: NewBorns, budget) -> Survivors:
        """Two-phase evaluation implementing GEPA Algorithm 1.
        
        This method is completely independent of the Generator's minibatch process.
        The Evaluator performs its own validation using its own minibatch sampling.
        """
        promoted_candidates = []
        
        for candidate in new_borns.candidates:
            is_promising = False
            
            if candidate.parents:
                # Phase 1: Independent minibatch validation against parent
                parent = candidate.parents[0]
                is_improved, cost = self._validate_on_minibatch(candidate, parent)
                budget.spend_on_evaluation(candidate.module, {"phase": "validation", "cost": cost})
                if is_improved:
                    is_promising = True
            else:
                # The initial candidate is always promising (no parent to compare against)
                is_promising = True

            # Phase 2: Full evaluation for promising candidates
            if is_promising:
                pareto_set = self.dataset_manager.get_pareto_set()
                candidate.evaluate_on_batch(pareto_set, self.metric)
                budget.spend_on_evaluation(
                    candidate.module, 
                    {"phase": "full_evaluation", "examples": len(pareto_set)}
                )
                promoted_candidates.append(candidate)

        logger.info(f"Two-phase evaluation: {len(promoted_candidates)}/{len(new_borns.candidates)} candidates promoted")
        return Survivors(*promoted_candidates, iteration=new_borns.iteration)

    def _validate_on_minibatch(self, child: 'Candidate', parent: 'Candidate') -> tuple[bool, int]:
        """
        Algorithm 1, Lines 13-14: Compare child and parent on validation minibatch.
        Gets a validation minibatch, separate from the generator's feedback minibatch.
        
        Args:
            child: Generated candidate to validate
            parent: Parent candidate for comparison
            
        Returns:
            Tuple of (is_improved: bool, cost: int)
        """
        # Gets a validation minibatch, separate from the generator's
        validation_minibatch = self.dataset_manager.get_validation_minibatch(self.minibatch_size)
        if not validation_minibatch: 
            return False, 0
        
        try:
            child_scores = [self._get_score(ex, child) for ex in validation_minibatch]
            parent_scores = [self._get_score(ex, parent) for ex in validation_minibatch]

            avg_child = sum(child_scores) / len(child_scores) if child_scores else 0
            avg_parent = sum(parent_scores) / len(parent_scores) if parent_scores else 0
            
            is_improved = avg_child > avg_parent
            cost = len(validation_minibatch) * 2  # Cost for evaluating both child and parent
            
            logger.debug(f"Minibatch validation: parent={avg_parent:.3f}, "
                        f"child={avg_child:.3f}, improved={is_improved}")
            
            return is_improved, cost
            
        except Exception as e:
            logger.warning(f"Minibatch validation failed: {e}")
            return False, len(validation_minibatch) * 2  # Still count the cost

    def _get_score(self, example, candidate):
        """Helper method to get score from a candidate on an example."""
        try:
            prediction = candidate.module(**example.inputs())
            return self.metric(example, prediction)
        except Exception as e:
            logger.warning(f"Failed to get score for candidate: {e}")
            return 0.0

    def get_metric(self) -> Callable:
        """Get the metric function used by this evaluator."""
        return self.metric
    
    def finish_compilation(self, result: dspy.Module) -> None:
        """Called when compilation ends."""
        pass
    
    def start_iteration(self, iteration: int, cohort, budget) -> None:
        """Called at the start of each iteration."""
        pass
    
    def finish_iteration(self, iteration: int, cohort, budget) -> None:
        """Called at the end of each iteration."""
        pass