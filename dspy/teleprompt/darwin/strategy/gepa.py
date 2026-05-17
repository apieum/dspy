"""GEPA - Default evolutionary optimization strategy."""

import logging
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

    def start_compilation(
        self, student: dspy.Module, *, trainset: list[dspy.Example], devset: list[dspy.Example] | None = None, teacher: dspy.Module | None = None, **kwargs
    ) -> None:
        """Initialize the strategy with the compilation parameters."""
        self.student = student
        self.trainset = trainset
        self.devset = devset if devset is not None else []
        self.teacher = teacher

        # Use dspy.DataLoader.train_test_split for internal train/validation split and dspy_uid generation
        from dspy.datasets.dataloader import DataLoader
        dl = DataLoader()
        splits = dl.train_test_split(self.trainset, train_size=1.0 - self.config.validation_split, random_state=self.config.seed)
        self.training_data, self.validation_data = splits['train'], splits['test']

        if self.config.verbose:
            logger.info(f"Data split: {len(self.training_data)} train, {len(self.validation_data)} validation, {len(self.devset)} test")

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
        if self.best_candidate:
            return Success(
                candidates=[self.best_candidate],
                history=None # TODO: Add history tracking
            )
        else:
            return Failure(
                message="Compilation failed, no candidate was found."
            )

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

        # Update best candidate tracking
        if self.current_survivors and not self.current_survivors.is_empty():
            best_in_generation = max(self.current_survivors.candidates,
                                   key=lambda c: c.average_score() if c.average_score() is not None else -1)

            if (self.best_candidate is None or
                (best_in_generation.average_score() is not None and self.best_candidate.average_score() is not None and
                 best_in_generation.average_score() > self.best_candidate.average_score())):
                self.best_candidate = best_in_generation
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

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

        # Generate new candidates
        self.current_newborns = self.generator.generate(self.current_parents, self.budget)

        # Cycle back to evaluation
        self.algorithm_state = "evaluate"
