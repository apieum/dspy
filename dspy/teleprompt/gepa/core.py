"""GEPA Framework - Strategy-based prompt optimization framework.

This module implements GEPA as a clean, configurable framework using
dependency injection and the strategy pattern for research flexibility.
"""

import logging
from typing import List, TYPE_CHECKING

import dspy
from dspy import Module

from .budget import Budget
from .selection import Selector
from .generation import Generator
from .evaluation import Evaluator
from .data.candidate import Candidate
from .data.cohort import Cohort,NewBorns, Survivors, Parents

if TYPE_CHECKING:
    from .dataset_manager import DatasetManagerFactory, DatasetManager

logger = logging.getLogger(__name__)


class GEPA:
    """Framework for prompt optimization with injectable strategies.

    GEPA orchestrates the optimization algorithm through clear phases:
    1. Scoring: Calculate candidate performance
    2. Filtering: Select surviving candidates
    3. Generation: Create new candidates
    4. Evaluation: Promote worthy new candidates
    5. Recurse to next generation

    All implementation details are delegated to injected strategies,
    making the framework highly configurable for research.
    """

    def __init__(self,
                 budget: Budget,
                 selector: Selector,
                 generator: Generator,
                 evaluator: Evaluator,
                 dataset_manager_factory: "DatasetManagerFactory",
                 patience: int = 3):
        """Initialize GEPA optimization with algorithms for each step.

        Args:
            budget: Manages optimization budget (LLM calls, iterations)
            selector: Selects surviving candidates based on scores
            generator: Creates new candidates via genetic operations
            evaluator: Promotes worthy new candidates (owns metric)
            dataset_manager_factory: Factory for creating dataset managers at compile time
            patience: Number of consecutive failed generations before termination (default: 3)
        """
        self.budget = budget
        self.selector = selector
        self.generator = generator
        self.evaluator = evaluator
        self.dataset_manager_factory = dataset_manager_factory
        self.current_iteration = 0

        # Patience mechanism for resilient termination
        self.patience = patience
        self.generations_without_progress = 0

    def start_compilation(self, student: Module, dataset_manager: "DatasetManager"):
        logger.info("Starting GEPA framework compilation...")
        # Pass the dataset manager to all components
        self.generator.start_compilation(student, dataset_manager)
        self.evaluator.start_compilation(student, dataset_manager)
        self.selector.start_compilation(student, dataset_manager)
        self.budget.start_compilation(student, dataset_manager)

    def finish_compilation(self, result: Module):
        logger.info("Terminating GEPA framework compilation.")
        self.budget.finish_compilation(result)
        self.selector.finish_compilation(result)
        self.generator.finish_compilation(result)
        self.evaluator.finish_compilation(result)

    def start_iteration(self, cohort: Cohort) -> None:
        """Starts a new iteration of the GEPA framework."""
        self.budget.start_iteration(self.current_iteration, cohort, self.budget)
        self.selector.start_iteration(self.current_iteration, cohort, self.budget)
        self.generator.start_iteration(self.current_iteration, cohort, self.budget)
        self.evaluator.start_iteration(self.current_iteration, cohort, self.budget)

    def finish_iteration(self, cohort: Cohort) -> None:
        self.budget.finish_iteration(self.current_iteration, cohort, self.budget)
        self.selector.finish_iteration(self.current_iteration, cohort, self.budget)
        self.generator.finish_iteration(self.current_iteration, cohort, self.budget)
        self.evaluator.finish_iteration(self.current_iteration, cohort, self.budget)

    def compile(self, student: Module, training_data: List[dspy.Example]) -> Module:
        """Main compilation method - entry point for optimization."""
        # Create dataset manager from factory with user's training data
        dataset_manager = self.dataset_manager_factory.create(training_data)
        self.start_compilation(student, dataset_manager)

        # Bootstrap: Create the first candidate (Gen 0).
        initial_candidate = Candidate(student.deepcopy(), generation_number=0)
        initial_newborns = NewBorns(initial_candidate, iteration=0)

        # The evaluator set scores on the parentless newborn.
        initial_survivors = self.evaluator.evaluate(initial_newborns, self.budget)

        # The selector promotes it to be the first parent pool.
        initial_parents = self.selector.promote(initial_survivors)

        # Start the main evolutionary loop.
        self.next_generation(initial_parents)

        # Return the best candidate from the final population.
        best_module = self.selector.best_candidate().module
        best_module._compiled = True
        self.finish_compilation(best_module)
        return best_module

    def next_generation(self, parents: Parents) -> Parents:
        """Recursive optimization step implementing the core algorithm."""
        self.current_iteration += 1
        logger.info(f"Processing generation {self.current_iteration} with {parents.size()} parents.")

        # Termination condition
        if self.budget <= 0:
            logger.info("Budget exhausted - terminating optimization.")
            return parents

        self.start_iteration(parents)

        # 1. Generator creates new candidates.
        new_borns = self.generator.generate(parents, self.budget)

        if new_borns.is_empty():
            self.generations_without_progress += 1
            logger.info(f"No new candidates generated (patience: {self.generations_without_progress}/{self.patience})")
            self.finish_iteration(parents)

            # Check if we've exceeded patience
            if self.generations_without_progress >= self.patience:
                logger.info(f"Patience exhausted after {self.patience} consecutive failed generations - terminating")
                return parents

            # Still have patience, try again with current parents
            return self.next_generation(parents)

        # 2. Evaluator handles the two-phase evaluation for the new children.
        new_survivors = self.evaluator.evaluate(new_borns, self.budget)

        # Check if any candidates survived evaluation
        if new_survivors.is_empty():
            self.generations_without_progress += 1
            logger.info(f"No candidates survived evaluation (patience: {self.generations_without_progress}/{self.patience})")
            self.finish_iteration(parents)

            # Check if we've exceeded patience
            if self.generations_without_progress >= self.patience:
                logger.info(f"Patience exhausted after {self.patience} consecutive failed generations - terminating")
                return parents

            # Still have patience, try again
            return self.next_generation(parents)

        # 3. Selector adds the successful survivors to the pool and returns the new frontier.
        next_gen_parents = self.selector.promote(new_survivors)

        # Reset patience counter - we made progress!
        if not new_survivors.is_empty():
            if self.generations_without_progress > 0:
                logger.info(f"Progress made! Resetting patience counter (was {self.generations_without_progress})")
            self.generations_without_progress = 0

        self.finish_iteration(next_gen_parents)
        return self.next_generation(next_gen_parents)

    def _show_results(self) -> Module:
        """Select and return the best candidate when optimization terminates."""
        logger.info("Selecting best candidate from final optimization results")

        best_candidate = self.selector.best_candidate()
        best_score = best_candidate.average_task_score()

        logger.info(f"Selected best candidate with score: {best_score:.4f}")

        compiled_module = best_candidate.module
        compiled_module._compiled = True
        return compiled_module

    @staticmethod
    def create_basic(metric, max_calls: int = 1000, population_size: int = 10, patience: int = 3) -> "GEPA":
        """Create a basic GEPA optimizer with standard settings."""
        from .budget import LMCallsBudget
        from .selection import ParetoFrontier
        from .generation import ReflectivePromptMutation
        from .generation.feedback import FeedbackProvider
        from .evaluation import GEPAEvaluator
        from .dataset_manager import DefaultDatasetManagerFactory

        return GEPA(
            budget=LMCallsBudget(max_calls),
            selector=ParetoFrontier(),
            generator=ReflectivePromptMutation(
                feedback_provider=FeedbackProvider(metric=metric)
            ),
            evaluator=GEPAEvaluator(
                metric=metric,
            ),
            dataset_manager_factory=DefaultDatasetManagerFactory(),
            patience=patience
        )


    @staticmethod
    def create_iteration_limited(metric, max_iterations: int = 20, population_size: int = 8, patience: int = 3) -> "GEPA":
        """Create a GEPA optimizer limited by iterations rather than LLM calls."""
        from .budget import IterationBudget
        from .selection import ParetoFrontier
        from .generation import ReflectivePromptMutation
        from .generation.feedback import FeedbackProvider
        from .evaluation import GEPAEvaluator
        from .dataset_manager import DefaultDatasetManagerFactory

        return GEPA(
            budget=IterationBudget(max_iterations),
            selector=ParetoFrontier(),
            generator=ReflectivePromptMutation(
                feedback_provider=FeedbackProvider(metric=metric)
            ),
            evaluator=GEPAEvaluator(
                metric=metric,
            ),
            dataset_manager_factory=DefaultDatasetManagerFactory(),
            patience=patience
        )

    @staticmethod
    def create_with_merge(metric, max_calls: int = 1000, population_size: int = 12, patience: int = 3) -> "GEPA":
        """Create GEPA optimizer with System-Aware Merge for complex optimization."""
        from .budget import LMCallsBudget
        from .selection import ParetoFrontier
        from .generation import SystemAwareMerge
        from .evaluation import GEPAEvaluator
        from .dataset_manager import DefaultDatasetManagerFactory

        return GEPA(
            budget=LMCallsBudget(max_calls),
            selector=ParetoFrontier(),
            generator=SystemAwareMerge(),
            evaluator=GEPAEvaluator(
                metric=metric,
            ),
            dataset_manager_factory=DefaultDatasetManagerFactory(),
            patience=patience
        )
