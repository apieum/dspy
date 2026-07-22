"""LM calls budget implementation."""

import logging
from typing import List, Dict, Any, Optional
import dspy
from .budget import Budget

logger = logging.getLogger(__name__)


class LMCallsBudget(Budget):
    """Budget that tracks LLM API calls."""

    def __init__(self, max_calls: int, evaluation_max_calls: Optional[int] = None,
                 generation_max_calls: Optional[int] = None):
        self.max_calls = max_calls
        self.evaluation_max_calls = evaluation_max_calls if evaluation_max_calls is not None else max_calls
        self.generation_max_calls = generation_max_calls if generation_max_calls is not None else max_calls
        self.consumed_calls = 0
        self.evaluation_calls = 0
        self.generation_calls = 0
        self.iteration_costs = []
        self.tracked_modules = {}  # module_id -> last_known_history_size

    def _spend(self, calls: int, phase: str) -> None:
        """Record budget consumption without exceeding the configured maximum."""
        calls = max(0, calls)
        if phase == "evaluation":
            allowed = min(calls, self.evaluation_max_calls - self.evaluation_calls)
            self.evaluation_calls += allowed
        else:
            allowed = min(calls, self.generation_max_calls - self.generation_calls)
            self.generation_calls += allowed
        self.consumed_calls = min(self.max_calls, self.consumed_calls + allowed)

    def spend_on_evaluation(self, module: dspy.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost based on actual expected evaluation calls, not module history."""
        # Use expected calls based on the evaluation phase and examples
        if module and hasattr(module, 'history') and module.history:
            new_calls = len(module.history) - self.tracked_modules.get(id(module), 0)
            self._spend(new_calls, "evaluation")
            self.tracked_modules[id(module)] = len(module.history)
            logger.debug(f"Evaluation cost: {new_calls} calls - {metadata}")
        elif metadata:
            if metadata.get('phase') == 'validation':
                # Validation should use minibatch size from metadata
                expected_calls = metadata.get('cost', 1)  # Use the cost from validation logic
                self._spend(expected_calls, "evaluation")
                logger.debug(f"Evaluation cost: {expected_calls} calls - {metadata}")
            elif metadata.get('phase') == 'full_evaluation':
                # Full evaluation should be exactly the number of examples
                expected_calls = metadata.get('examples', 1)
                self._spend(expected_calls, "evaluation")
                logger.debug(f"Evaluation cost: {expected_calls} calls - {metadata}")
            elif metadata.get('phase') == 'test':
                expected_calls = metadata.get('examples', 1)
                self._spend(expected_calls, "evaluation")
                logger.debug(f"Evaluation cost: {expected_calls} calls - {metadata}")

    def spend_on_generation(self, module: Optional[dspy.Module] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of generating new candidates - typically 1 LLM call for reflection."""
        generation_cost = 1  # Standard cost for one reflection/mutation LLM call
        self._spend(generation_cost, "generation")
        logger.debug(f"Generation cost: {generation_cost} calls - {metadata}")

    def get_remaining(self) -> dict:
        remaining_calls = max(0, self.max_calls - self.consumed_calls)
        return {
            "calls": remaining_calls,
            "evaluation_calls": max(0, self.evaluation_max_calls - self.evaluation_calls),
            "generation_calls": max(0, self.generation_max_calls - self.generation_calls),
            "percentage": (remaining_calls / self.max_calls) * 100 if self.max_calls > 0 else 0
        }

    def can_spend(self, phase: str, calls: int = 1) -> bool:
        """Return whether a phase has room for the requested cost."""
        if calls < 0:
            raise ValueError("calls must be non-negative")
        if phase == "evaluation":
            remaining = self.evaluation_max_calls - self.evaluation_calls
        elif phase == "generation":
            remaining = self.generation_max_calls - self.generation_calls
        else:
            raise ValueError(f"Unknown budget phase: {phase}")
        return calls <= remaining and calls <= self.max_calls - self.consumed_calls

    def __int__(self) -> int:
        return self.max_calls - self.consumed_calls

    def __lt__(self, other: int) -> bool:
        return (self.max_calls - self.consumed_calls) < other
