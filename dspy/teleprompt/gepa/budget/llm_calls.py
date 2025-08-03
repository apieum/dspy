"""LLM calls budget implementation."""

import logging
from typing import List
import dspy
from .budget import Budget

logger = logging.getLogger(__name__)


class LLMCallsBudget(Budget):
    """Budget that tracks LLM API calls."""
    
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.consumed_calls = 0
        self.iteration_costs = []
        
    def can_afford(self, cost: int, operation_type: str) -> bool:
        return self.consumed_calls + cost <= self.max_calls
        
    def consume(self, cost: int, operation_type: str) -> None:
        self.consumed_calls += cost
        
    def is_empty(self) -> bool:
        return self.consumed_calls >= self.max_calls
        
    def get_remaining(self) -> int:
        return max(0, self.max_calls - self.consumed_calls)
    
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