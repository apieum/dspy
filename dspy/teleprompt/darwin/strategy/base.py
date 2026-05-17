"""Base class for all optimization strategies."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING, TypeVar, Generic

import dspy
from ..data.candidate import Candidate
from ..result import Result

if TYPE_CHECKING:
    from ..config import DarwinConfig
    from ..budget import Budget
    from ..selection import Selector
    from ..generation import Generator
    from ..evaluation import Evaluator

R = TypeVar("R", bound=Result)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC, Generic[R]):
    """Evolutionary algorithm strategy that owns components and makes all tactical decisions.

    The strategy acts as the brain of the evolutionary algorithm:
    - Owns and instantiates all evolutionary operators (selector, generator, evaluator, budget)
    - Makes all tactical decisions about when to select, mutate, crossover, evaluate
    - Manages population, data splitting, and evolutionary state
    - Returns callables for the optimizer to execute

    Components never know about the strategy - strategy calls them with data.
    """

    def __init__(self, config: 'DarwinConfig'):
        """Initialize strategy with configuration parameters."""
        self.config = config

        # Component instances (lazy instantiation)
        self._budget: Optional['Budget'] = None
        self._selector: Optional['Selector'] = None
        self._generator: Optional['Generator'] = None
        self._evaluator: Optional['Evaluator'] = None

        # Evolutionary state
        self.current_generation = 0
        self.best_candidate: Optional[Candidate] = None
        self.generations_without_improvement = 0

        # Data management
        self.trainset: List[dspy.Example] = []
        self.devset: List[dspy.Example] = []
        self.training_data: List[dspy.Example] = []
        self.validation_data: List[dspy.Example] = []

    @abstractmethod
    def start_compilation(
        self, student: dspy.Module, *, trainset: list[dspy.Example], devset: list[dspy.Example] | None = None, teacher: dspy.Module | None = None, **kwargs
    ) -> None:
        """Initialize the strategy with the compilation parameters."""
        raise NotImplementedError

    @abstractmethod
    def next_step(self) -> bool:
        """Perform one step of the optimization process.

        Returns:
            True if there are more steps, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def terminate_compilation(self) -> R:
        """Get the result of the optimization process.

        Returns:
            A Result object.
        """
        raise NotImplementedError

    def _create_minibatch(self, data: List[dspy.Example], size: int) -> List[dspy.Example]:
        """Create a minibatch from validation data using intelligent sampling."""
        if not data:
            return []

        # If we have fewer examples than requested size, return all
        if len(data) <= size:
            return data.copy()

        # Use random sampling to get diverse examples for minibatch
        # This is better than just taking first N examples
        import random
        return random.sample(data, size)

    # Component property accessors (lazy instantiation)
    @property
    def budget(self) -> 'Budget':
        """Get budget component, instantiating if needed."""
        if self._budget is None:
            from ..budget import Budget
            self._budget = self.config.budget(max_calls=self.config.max_lm_calls)
        return self._budget

    @property
    def selector(self) -> 'Selector':
        """Get selection component, instantiating if needed."""
        if self._selector is None:
            from ..selection import Selector
            self._selector = self.config.selection()
        return self._selector

    @property
    def generator(self) -> 'Generator':
        """Get mutation generator component, instantiating if needed."""
        if self._generator is None:
            # Strategy decides what metric and data to pass to generator
            from ..generation.feedback import FeedbackProvider
            from ..generation import SystemAwareMerge

            feedback_provider = FeedbackProvider(assessor=self.config.enhanced_feedback)
            # Strategy provides intelligently sampled minibatch for feedback
            feedback_data = self._create_minibatch(self.validation_data, self.config.minibatch_size)
            if self.config.mutation is SystemAwareMerge:
                self._generator = self.config.mutation(assessor=self.config.enhanced_feedback)
                self._generator.start_compilation(self.student, verbose=self.config.verbose, feedback_data=feedback_data)
            else:
                self._generator = self.config.mutation(
                    feedback_provider=feedback_provider,
                    feedback_data=feedback_data
                )
        return self._generator

    @property
    def evaluator(self) -> 'Evaluator':
        """Get evaluation component, instantiating if needed."""
        if self._evaluator is None:
            # Strategy creates appropriate data splits for different evaluation phases
            minibatch_data = self._create_minibatch(self.validation_data, self.config.minibatch_size)
            self._evaluator = self.config.evaluation(
                assessor=self.config.fitness_function,
                minibatch_data=minibatch_data,         # Intelligently sampled minibatch for quick validation
                validation_data=self.validation_data  # Full set for comprehensive evaluation
            )
        return self._evaluator

    # Evolutionary decision methods
    def should_terminate(self) -> bool:
        """Decide if optimization should terminate."""
        # Check budget exhaustion
        if self.budget <= 0:
            return True

        # Check patience (generations without improvement)
        if self.generations_without_improvement >= self.config.patience:
            return True

        return False
