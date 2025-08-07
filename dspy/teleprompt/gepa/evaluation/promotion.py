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

    def evaluate(self, cohort: NewBorns, budget: Budget) -> Survivors:
        """
        Two-phase evaluation from GEPA Algorithm 1.
        
        For candidates with parents: minibatch validation + full evaluation
        For candidates without parents: direct full evaluation
        
        Args:
            cohort: Newly generated candidates to evaluate
            budget: Budget to track evaluation costs
            
        Returns:
            Survivors containing only validated candidates with full task_scores
        """
        validated_candidates = []

        for candidate in cohort.candidates:
            if not candidate.parents:
                # No parent comparison needed - directly evaluate on pareto data
                logger.debug("Evaluating parentless candidate directly on pareto data")
                pareto_set = self.dataset_manager.get_pareto_set()
                candidate.evaluate_on_batch(pareto_set, self.metric)
                budget.spend_on_evaluation(candidate.module, {
                    "phase": "direct_pareto_evaluation", 
                    "examples": len(pareto_set)
                })
                validated_candidates.append(candidate)
            else:
                parent = candidate.parents[0]  # Assuming single parent for mutation

                # Phase 1: Minibatch Validation
                is_improved, minibatch_cost = self._validate_on_minibatch(candidate, parent)
                budget.spend_on_evaluation(candidate.module, {
                    "phase": "minibatch_validation", 
                    "cost": minibatch_cost,
                    "examples": minibatch_cost // 2
                })

                if is_improved:
                    # Phase 2: Full Evaluation on Pareto Data
                    logger.debug(f"Candidate validated, running full Pareto evaluation")
                    pareto_set = self.dataset_manager.get_pareto_set()
                    candidate.evaluate_on_batch(pareto_set, self.metric)
                    budget.spend_on_evaluation(candidate.module, {
                        "phase": "full_pareto_evaluation", 
                        "examples": len(pareto_set)
                    })
                    validated_candidates.append(candidate)
                    logger.debug(f"Candidate promoted with average score: {candidate.average_task_score():.3f}")
                else:
                    logger.debug("Candidate did not improve on minibatch validation")

        logger.info(f"Two-phase evaluation: {len(validated_candidates)}/{len(cohort.candidates)} candidates validated")
        
        return Survivors(*validated_candidates, iteration=cohort.iteration)

    def _validate_on_minibatch(self, child, parent) -> tuple[bool, int]:
        """
        Algorithm 1, Lines 13-14: Compare child and parent on validation minibatch.
        
        Args:
            child: Generated candidate to validate
            parent: Parent candidate for comparison
            
        Returns:
            Tuple of (is_improved: bool, cost: int)
        """
        validation_minibatch = self.dataset_manager.get_validation_minibatch(self.minibatch_size)
        if not validation_minibatch:
            logger.warning("No validation data available for minibatch validation")
            return False, 0
        
        try:
            # Evaluate child on validation minibatch
            child_scores = []
            for example in validation_minibatch:
                prediction = child.module(**example.inputs())
                score = self.metric(example, prediction)
                child_scores.append(score)
            
            # Evaluate parent on same validation minibatch
            parent_scores = []
            for example in validation_minibatch:
                prediction = parent.module(**example.inputs())
                score = self.metric(example, prediction)
                parent_scores.append(score)

            # Calculate averages
            avg_child_score = sum(child_scores) / len(child_scores) if child_scores else 0
            avg_parent_score = sum(parent_scores) / len(parent_scores) if parent_scores else 0
            
            # Cost is number of examples * 2 (for child and parent)
            cost = len(validation_minibatch) * 2
            
            # Check if child better than parent
            is_improved = avg_child_score > avg_parent_score
            logger.debug(f"Minibatch validation: parent={avg_parent_score:.3f}, "
                        f"child={avg_child_score:.3f}, improved={is_improved}")
            
            return is_improved, cost
            
        except Exception as e:
            logger.warning(f"Minibatch validation failed: {e}")
            return False, len(validation_minibatch) * 2  # Still count the cost

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