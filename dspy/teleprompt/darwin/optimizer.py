"""Darwin Toolkit - Simple strategy executor for evolutionary optimization.

Darwin acts as a simple executor that applies evolutionary optimization strategies.
It takes a strategy class and configuration, then executes the strategy's decisions.
"""

import logging
from typing import TYPE_CHECKING, Type

import dspy
from dspy import Module

from dspy.teleprompt.teleprompt import Teleprompter
from .result import Result

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

    def compile(
        self,
        student: Module,
        *,
        trainset: list[dspy.Example],
        teacher: dspy.Module | None = None,
        valset: list[dspy.Example] | None = None,
        devset: list[dspy.Example] | None = None,
        **kwargs,
    ) -> dspy.Module:
        """Main compilation method - simple strategy execution loop.

        Args:
            student: DSPy module to optimize
            trainset: Training examples
            devset: Optional test set for final evaluation
        """
        compiled = self.strategy.compile(
            student,
            trainset=trainset,
            teacher=teacher,
            valset=valset,
            devset=devset,
            **kwargs,
        )
        self.latest_result = self.strategy.get_last_result()
        return compiled

    def get_last_result(self) -> Result | None:
        """Return the last result from the optimizer."""
        return self.latest_result
