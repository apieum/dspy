"""GEPA - Default evolutionary optimization strategy."""

import logging
import inspect
from typing import List, Optional, TYPE_CHECKING

import dspy
from .base import BaseStrategy
from ..data.candidate import Candidate
from ..data.cohort import NewBorns, Survivors, Parents
from ..result import Result, Success, Failure

if TYPE_CHECKING:
    from ..config import DarwinConfig

logger = logging.getLogger(__name__)


class GEPAStrategy(BaseStrategy[Result]):
    """Default implementation of evolutionary optimization strategy.

    Implements a simple evolutionary algorithm with the following steps:
    1. Initialize population with single candidate
    2. Evaluate candidates
    3. Select survivors
    4. Generate new candidates (mutation/crossover)
    5. Repeat until termination criteria met
    """

    def __init__(self, config: 'DarwinConfig'):
        super().__init__(config)
        self.algorithm_state = "initialize"  # initialize -> evaluate -> select -> generate -> repeat
        self.current_newborns: Optional[NewBorns] = None
        self.current_survivors: Optional[Survivors] = None
        self.current_parents: Optional[Parents] = None
        self._iteration_started = False
        self.history = []

    def start_compilation(
        self, student: dspy.Module, *, trainset: list[dspy.Example], devset: list[dspy.Example] | None = None, teacher: dspy.Module | None = None, **kwargs
    ) -> None:
        """Initialize the strategy with the compilation parameters."""
        self.student = student
        self.trainset = trainset
        self.devset = devset if devset is not None else []
        self.teacher = teacher
        self.current_generation = 0
        self.best_candidate = None
        self.generations_without_improvement = 0
        self.history = []

        # Centralize train/dev handling so experiments can inject a different
        # dataset policy without changing the strategy itself.
        manager_factory = self.config.dataset_manager_factory
        # Accept both the default factory class and an already-configured
        # factory instance.  The latter is useful when an experiment needs a
        # custom split policy or a stateful dataset manager.
        if isinstance(manager_factory, type):
            factory_parameters = inspect.signature(manager_factory).parameters
            factory_kwargs = {"split_ratio": self.config.validation_split}
            if "seed" in factory_parameters:
                factory_kwargs["seed"] = self.config.seed
            manager_factory = manager_factory(**factory_kwargs)
        manager = manager_factory.create(
            self.trainset,
            self.devset if self.devset else None,
        )
        self.dataset_manager = manager
        self.training_data = list(manager.get_eval_set().values())
        self.validation_data = list(
            manager.get_validation_minibatch(manager.num_dev_examples).values()
        )

        if self.config.verbose:
            logger.info(f"Data split: {len(self.training_data)} train, {len(self.validation_data)} validation, {len(self.devset)} test")

        self._notify("start_compilation", self.student, self.dataset_manager)

        # Initialize the first candidate
        initial_candidate = Candidate(self.student.deepcopy(), generation_number=0)
        self.current_newborns = NewBorns([initial_candidate], iteration=0)

        self.algorithm_state = "evaluate"  # Start with evaluation of the initial candidate

    def next_step(self) -> bool:
        """Implement the evolutionary algorithm state machine."""
        if self.should_terminate():
            return False

        if self.algorithm_state == "evaluate":
            self._evaluate_step()
        elif self.algorithm_state == "select":
            self._select_step()
        elif self.algorithm_state == "generate":
            self._generate_step()
        else:
            # Invalid state, terminate
            return False
        
        return True

    def terminate_compilation(self) -> Result:
        """Get the result of the optimization process."""
        final_candidate = self._select_final_candidate()
        self.best_candidate = final_candidate
        if self.best_candidate:
            result = Success(
                candidates=[self.best_candidate],
                history=list(self.history),
            )
        else:
            result = Failure(
                reason="Compilation failed, no candidate was found."
            )
        result_module = self.best_candidate.module if self.best_candidate is not None else self.student
        if self.best_candidate is not None:
            result_module._compiled = True
        self._notify("finish_compilation", result_module)
        return result

    def _select_final_candidate(self) -> Optional[Candidate]:
        """Select the final candidate from the accumulated Pareto state.

        The selector has the global task-level view required by GEPA.  Keep
        the strategy's best-per-generation candidate as a fallback for custom
        selectors that do not expose a final-candidate method.
        """
        candidate = self.best_candidate
        selector_best = getattr(self.selector, "best_candidate", None)
        if callable(selector_best):
            try:
                selected = selector_best()
                if isinstance(selected, Candidate):
                    candidate = selected
            except (RuntimeError, ValueError):
                # A selector may have no accumulated scores when compilation
                # ends before the first evaluation completes.
                pass
        return candidate

    def _evaluate_step(self):
        """Evaluate current candidates."""
        if self.config.verbose:
            logger.info(f"Evaluating candidates in generation {self.current_generation}")

        if self.current_newborns is None or self.current_newborns.is_empty():
            # No candidates to evaluate, move to selection
            self.algorithm_state = "select"
            return

        # Evaluate the newborns (including initial candidate)
        self.current_survivors = self.evaluator.evaluate(self.current_newborns, self.budget)

        generation_best_score = None

        # Update best candidate tracking
        if self.current_survivors and not self.current_survivors.is_empty():
            best_in_generation = max(self.current_survivors.candidates,
                                   key=lambda c: c.average_score() if c.average_score() is not None else -1)
            generation_best_score = best_in_generation.average_score()

            if (self.best_candidate is None or
                (best_in_generation.average_score() is not None and self.best_candidate.average_score() is not None and
                 best_in_generation.average_score() > self.best_candidate.average_score())):
                self.best_candidate = best_in_generation
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

        self.history.append({
            "generation": self.current_generation,
            "evaluated_candidates": len(self.current_newborns),
            "surviving_candidates": len(self.current_survivors),
            "best_score": generation_best_score,
        })

        self.algorithm_state = "select"

    def _select_step(self):
        """Select candidates for next generation."""
        if self.config.verbose:
            logger.info(f"Selecting candidates for generation {self.current_generation + 1}")

        if self.current_survivors is None or self.current_survivors.is_empty():
            # No survivors to select from, move to generate
            self.algorithm_state = "generate"
            return

        # Promote survivors to parents for next generation
        self.current_parents = self.selector.promote(self.current_survivors)
        if self._iteration_started:
            self._notify("finish_iteration", self.current_generation, self.current_parents, self.budget)
            self._iteration_started = False
        self.algorithm_state = "generate"

    def _generate_step(self):
        """Generate new candidates."""
        self.current_generation += 1

        if self.config.verbose:
            logger.info(f"Generating candidates for generation {self.current_generation}")

        if self.current_parents is None or self.current_parents.is_empty():
            # No parents available, terminate by setting state that leads to exit
            self.algorithm_state = "terminate"
            return

        self._notify("start_iteration", self.current_generation, self.current_parents, self.budget)
        self._iteration_started = True

        # Generate new candidates
        self.current_newborns = self.generator.generate(self.current_parents, self.budget)

        # Cycle back to evaluation
        self.algorithm_state = "evaluate"
