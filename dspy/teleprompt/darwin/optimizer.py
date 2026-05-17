"""Darwin Toolkit - Simple strategy executor for evolutionary optimization.

Darwin acts as a simple executor that applies evolutionary optimization strategies.
It takes a strategy class and configuration, then executes the strategy's decisions.
"""

import logging
from typing import TYPE_CHECKING, Type

import dspy
from dspy import Module

from dspy.teleprompt.teleprompt import Teleprompter
from .result import Failure, OptimizationFailureError, Result

if TYPE_CHECKING:
    from .config import DarwinConfig
    from .strategy import BaseStrategy

logger = logging.getLogger(__name__)


class Darwin(Teleprompter):
    """Simple strategy executor for evolutionary optimization.

    Darwin's responsibility is minimal:
    1. Take a strategy class and configuration
    2. Instantiate the strategy with the configuration
    3. Execute strategy decisions by calling next_step() in a loop
    4. Return the final result

    All evolutionary logic is delegated to the strategy.
    """

    def __init__(self, strategy_class: Type['BaseStrategy'], config: 'DarwinConfig') -> None:
        """Initialize Darwin with strategy class and configuration.

        Args:
            strategy_class: Class of strategy to instantiate and execute
            config: Configuration with evolutionary parameters and component classes
        """
        super().__init__()
        self.strategy = strategy_class(config)
        self.config = config
        self.latest_result: Result | None = None

    def compile(self, student: Module, *, trainset: list[dspy.Example], devset: list[dspy.Example] | None = None, teacher: dspy.Module | None = None, **kwargs) -> dspy.Module:
        """Main compilation method - simple strategy execution loop.

        Args:
            student: DSPy module to optimize
            trainset: Training examples
            devset: Optional test set for final evaluation
        """
        try:
            self.strategy.start_compilation(student=student, trainset=trainset, devset=devset, teacher=teacher, **kwargs)

            if self.config.verbose:
                pass

            while self.strategy.next_step():
                pass

            result_obj = self.strategy.terminate_compilation()
            self.latest_result = result_obj

            if isinstance(result_obj, Failure):
                raise OptimizationFailureError(f"Optimization failed: {result_obj.message}")

            best_candidate = result_obj.get_best_generalist()
            if best_candidate and best_candidate.module:
                best_candidate.module._compiled = True
                return best_candidate.module

            raise OptimizationFailureError("Optimization failed: no compiled module was produced")

        except Exception as e:
            raise OptimizationFailureError(str(e))

    def get_last_result(self) -> Result | None:
        """Return the last result from the optimizer."""
        return self.latest_result
