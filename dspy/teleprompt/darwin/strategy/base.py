"""Base class for all optimization strategies."""

import logging
import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING, TypeVar, Generic

import dspy
from ..data.candidate import Candidate
from ..result import Result
from ..compilation_observer import CompilationObserver
from ..dataset_manager import DatasetManager

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
        self.dataset_manager: Optional[DatasetManager] = None
        self.observers: tuple[CompilationObserver, ...] = tuple(config.observers)

    def _notify(self, event: str, *args) -> None:
        """Notify opt-in lifecycle observers without coupling components to them."""
        for observer in self.observers:
            callback = getattr(observer, event, None)
            if callback is not None:
                callback(*args)

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
            self._budget = self._instantiate_budget()
        return self._budget

    @property
    def selector(self) -> 'Selector':
        """Get selection component, instantiating if needed."""
        if self._selector is None:
            from ..selection import Selector
            self._selector = self.config.selection()
            configure = getattr(self._selector, "configure", None)
            if callable(configure):
                configure(self.config)
        return self._selector

    @property
    def generator(self) -> 'Generator':
        """Get mutation generator component, instantiating if needed."""
        if self._generator is None:
            self._generator = self._instantiate_generator(self.config.mutation)
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
                validation_data=self.validation_data,  # Full set for comprehensive evaluation
                acceptance_criterion=(
                    self.config.acceptance_criterion()
                    if isinstance(self.config.acceptance_criterion, type)
                    else self.config.acceptance_criterion
                ),
                evaluation_cache=self.evaluation_cache,
                proposal_selection=(
                    self.config.proposal_selection()
                    if isinstance(self.config.proposal_selection, type)
                    else self.config.proposal_selection
                ),
                validation_policy=(
                    self.config.validation_policy()
                    if isinstance(self.config.validation_policy, type)
                    else self.config.validation_policy
                ),
            )
            self._evaluator.start_compilation(
                getattr(self, "student", None),
                dataset_manager=self.dataset_manager,
                verbose=self.config.verbose,
            )
        return self._evaluator

    # Evolutionary decision methods
    def should_terminate(self) -> bool:
        """Decide if optimization should terminate."""
        # Check budget exhaustion
        if self.budget <= 0:
            if not getattr(self, "_budget_exhaustion_notified", False):
                self._notify("budget_exhausted", self.budget)
                self._budget_exhaustion_notified = True
            return True

        # Check patience (generations without improvement)
        if self.generations_without_improvement >= self.config.patience:
            return True

        # Generation zero is the seed evaluation. Stop before creating a new
        # generation once the configured number of mutation rounds is done.
        if getattr(self, "algorithm_state", None) == "generate" and self.current_generation >= self.config.max_iterations:
            return True

        return False

    def _instantiate_budget(self) -> 'Budget':
        budget_cls = self.config.budget
        signature = inspect.signature(budget_cls)
        kwargs = {}
        if "max_calls" in signature.parameters:
            kwargs["max_calls"] = self.config.max_lm_calls
            if "evaluation_max_calls" in signature.parameters:
                kwargs["evaluation_max_calls"] = self.config.max_evaluation_calls
            if "generation_max_calls" in signature.parameters:
                kwargs["generation_max_calls"] = self.config.max_generation_calls
        elif "max_iterations" in signature.parameters:
            kwargs["max_iterations"] = self.config.max_iterations
        elif "total_budget" in signature.parameters:
            kwargs["total_budget"] = self.config.max_lm_calls
        return budget_cls(**kwargs)

    def _instantiate_generator(self, generator_factory) -> 'Generator':
        from ..generation.feedback import FeedbackProvider

        mutation_config = self.config.mutation_config
        feedback_assessor = (
            mutation_config.feedback_provider.assessor
            if mutation_config and mutation_config.feedback_provider is not None
            else self.config.enhanced_feedback or self.config.fitness_function
        )
        feedback_function = mutation_config.enhanced_feedback_function if mutation_config else None
        feedback_provider = (
            mutation_config.feedback_provider
            if mutation_config and mutation_config.feedback_provider is not None
            else FeedbackProvider(assessor=feedback_assessor, feedback_function=feedback_function)
        )
        feedback_data = self._create_minibatch(
            self.validation_data,
            mutation_config.minibatch_size if mutation_config else self.config.minibatch_size,
        )

        kwargs = {
            "feedback_provider": feedback_provider,
            "feedback_data": feedback_data,
            "assessor": feedback_assessor,
            "config": mutation_config,
        }
        if mutation_config:
            kwargs.update(
                {
                    "reflection_strategy": mutation_config.reflection_strategy,
                    "module_selection": mutation_config.module_selection_strategy.value,
                    "max_retries": mutation_config.max_retries,
                }
            )

        generator = generator_factory(**kwargs)

        student = getattr(self, "student", None)
        if student is not None:
            generator.start_compilation(
                student,
                dataset_manager=self.dataset_manager,
                feedback_data=feedback_data,
                verbose=self.config.verbose,
            )
        return generator
