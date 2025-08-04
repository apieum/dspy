"""LLM calls budget implementation."""

import logging
from typing import List, Dict, Any, Optional
import dspy
from .budget import Budget

logger = logging.getLogger(__name__)


class LLMCallsBudget(Budget):
    """Budget that tracks LLM API calls."""
    
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.consumed_calls = 0
        self.iteration_costs = []
        
        
    def spend_on_evaluation(self, module: dspy.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of evaluating a candidate module by counting LLM calls from history."""
        if hasattr(module, 'history') and module.history:
            # Count new calls since last check (simplified approach)
            new_calls = len(module.history)
            self.consumed_calls += new_calls
            if metadata:
                logger.debug(f"Evaluation cost: {new_calls} calls - {metadata}")
        
    def spend_on_generation(self, module: Optional[dspy.Module] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of generating new candidates - typically 1 LLM call for reflection."""
        generation_cost = 1  # Standard cost for one reflection/mutation LLM call
        self.consumed_calls += generation_cost
        if metadata:
            logger.debug(f"Generation cost: {generation_cost} calls - {metadata}")
        
    def get_remaining(self) -> dict:
        remaining_calls = max(0, self.max_calls - self.consumed_calls)
        return {
            "calls": remaining_calls,
            "percentage": (remaining_calls / self.max_calls) * 100 if self.max_calls > 0 else 0
        }
    
    
    # CompilationObserver lifecycle methods
    
    def start_compilation(self, student: dspy.Module, training_data: List[dspy.Example]) -> None:
        """Initialize budget tracking when compilation begins."""
        logger.info(f"Starting compilation with budget of {self.max_calls} LLM calls")
        self.consumed_calls = 0
        self.iteration_costs = []
    
    def finish_compilation(self, result: dspy.Module, final_pool) -> None:
        """Log final budget usage when compilation ends."""
        usage_percentage = (self.consumed_calls / self.max_calls) * 100
        logger.info(f"Compilation complete - Used {self.consumed_calls}/{self.max_calls} calls ({usage_percentage:.1f}%)")
        if self.iteration_costs:
            avg_per_iteration = sum(self.iteration_costs) / len(self.iteration_costs)
            logger.info(f"Average cost per iteration: {avg_per_iteration:.1f} calls")
    
    def start_iteration(self, iteration: int, cohort, budget) -> None:
        """Track iteration start."""
        self._iteration_start_calls = self.consumed_calls
        
    def finish_iteration(self, iteration: int, filtered_cohort, budget) -> None:
        """Track iteration cost."""
        iteration_cost = self.consumed_calls - self._iteration_start_calls
        self.iteration_costs.append(iteration_cost)
        remaining = self.get_remaining()
        logger.debug(f"Iteration {iteration}: Used {iteration_cost} calls, {remaining} remaining")