"""LM calls budget implementation."""

import logging
from typing import List, Dict, Any, Optional
import dspy
from .budget import Budget

logger = logging.getLogger(__name__)


class LMCallsBudget(Budget):
    """Budget that tracks LLM API calls."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.consumed_calls = 0
        self.iteration_costs = []
        self.tracked_modules = {}  # module_id -> last_known_history_size


    def spend_on_evaluation(self, module: dspy.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost based on actual expected evaluation calls, not module history."""
        # Use expected calls based on the evaluation phase and examples
        if metadata:
            if metadata.get('phase') == 'validation':
                # Validation should use minibatch size from metadata
                expected_calls = metadata.get('cost', 1)  # Use the cost from validation logic
            elif metadata.get('phase') == 'full_evaluation':
                # Full evaluation should be exactly the number of examples
                expected_calls = metadata.get('examples', 1)
            elif metadata.get('phase') == 'evaluation' and module and hasattr(module, 'history'):
                # Legacy test pattern: count actual calls from module history
                expected_calls = len(module.history) if module.history else 0
            else:
                expected_calls = 1
        else:
            expected_calls = 0 if not module or not hasattr(module, 'history') or not module.history else len(module.history)

        self.consumed_calls += expected_calls
        logger.debug(f"Evaluation cost: {expected_calls} calls - {metadata}")

    def spend_on_generation(self, module: Optional[dspy.Module] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of generating new candidates - typically 1 LLM call for reflection."""
        generation_cost = 1  # Standard cost for one reflection/mutation LLM call
        self.consumed_calls += generation_cost
        logger.debug(f"Generation cost: {generation_cost} calls - {metadata}")

    def get_remaining(self) -> dict:
        remaining_calls = max(0, self.max_calls - self.consumed_calls)
        return {
            "calls": remaining_calls,
            "percentage": (remaining_calls / self.max_calls) * 100 if self.max_calls > 0 else 0
        }

    def peek(self) -> int:
        """Return remaining budget without consuming it."""
        return max(0, self.max_calls - self.consumed_calls)

    def __le__(self, other) -> bool:
        """Allow budget comparison for while loops."""
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.peek() <= other

    def __gt__(self, other) -> bool:
        """Allow budget comparison for while loops."""
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.peek() > other


    # CompilationObserver lifecycle methods

    def start_compilation(self, student: dspy.Module, split_strategy=None, verbose: bool=False) -> None:
        """Initialize budget tracking when compilation begins."""
        logger.info(f"Starting compilation with budget of {self.max_calls} LLM calls")
        if split_strategy:
            feedback_size = len(split_strategy.internal_validation_set)
            logger.info(f"Internal validation examples: {feedback_size}")
        else:
            logger.info("No split strategy provided")
        self.consumed_calls = 0
        self.iteration_costs = []

    def finish_compilation(self, result: dspy.Module) -> None:
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
