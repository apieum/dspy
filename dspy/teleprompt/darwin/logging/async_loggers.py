"""Specialized async logger observers for Darwin framework.

These observers handle all logging and debugging output through the async ChannelContext system,
providing non-blocking, efficient logging that doesn't slow down the optimization algorithm.
"""

import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from typing import Protocol
from ..data.candidate import Candidate
from ..data.cohort import Cohort, NewBorns, Survivors, Parents


class AsyncLogger:
    """Base async logger that routes messages through the ChannelContext system."""

    def __init__(self, logger_name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.level = level

    def _should_log(self, level: int) -> bool:
        """Check if message should be logged based on level."""
        return self.logger.isEnabledFor(level) and level >= self.level


class EvaluationLogger(AsyncLogger):
    """Specialized logger for evaluation operations."""

    def __init__(self, verbose: bool = False):
        super().__init__("darwin.evaluation", logging.DEBUG if verbose else logging.INFO)
        self.verbose = verbose

    async def validate_on_minibatch(self, child: Candidate, minibatch_size: int) -> None:
        """Log validation phase information."""
        if not self._should_log(logging.DEBUG):
            return

        # Extract instruction information efficiently
        try:
            from dspy.teleprompt.utils import get_signature

            # Log child instruction
            child_predictors = child.module.predictors()
            if child_predictors:
                child_instruction = get_signature(child_predictors[0]).instructions
                self.logger.info(f"VALIDATION INSTRUCTION DEBUG: child instruction: {child_instruction}")

            # Log parent instructions
            for i, parent in enumerate(child.parents):
                parent_predictors = parent.module.predictors()
                if parent_predictors:
                    parent_instruction = get_signature(parent_predictors[0]).instructions
                    self.logger.info(f"VALIDATION INSTRUCTION DEBUG: parent {i} instruction: {parent_instruction}")

            # Log validation phase start
            self.logger.debug(f"Validation phase: evaluating on {minibatch_size} examples from internal validation set")

        except Exception as e:
            self.logger.warning(f"Failed to log validation details: {e}")

    async def validation_result(self, child: Candidate, result_data: dict) -> None:
        """Log validation result."""
        if self._should_log(logging.INFO):
            passed = result_data['passed']
            cost = result_data['cost']
            parent_avg = result_data['parent_avg']
            child_avg = result_data['child_avg']
            self.logger.info(f"Validation result: parent={parent_avg:.3f}, child={child_avg:.3f}, passes_filter={passed}")
            if self._should_log(logging.DEBUG):
                self.logger.debug(f"Validation cost: {cost}")

    async def parent_fast_compare_summary(self, summary_data: dict) -> None:
        """Log ParentFastCompare summary."""
        if self._should_log(logging.INFO):
            passed = summary_data['passed']
            total = summary_data['total']
            self.logger.info(f"ParentFastCompare: {passed}/{total} candidates passed validation.")

    async def comprehensive_evaluation_start(self, eval_data: dict) -> None:
        """Log comprehensive evaluation start."""
        if self._should_log(logging.DEBUG):
            candidates_count = eval_data['candidates_count']
            tasks_count = eval_data['tasks_count']
            self.logger.debug(f"Full evaluation phase: assessing {candidates_count} candidates on {tasks_count} examples")

    async def comprehensive_evaluation_complete(self, eval_data: dict) -> None:
        """Log comprehensive evaluation completion."""
        if self._should_log(logging.INFO):
            candidates_count = eval_data['candidates_count']
            tasks_count = eval_data['tasks_count']
            self.logger.info(f"Comprehensive evaluation completed for {candidates_count} candidates on {tasks_count} tasks")

    async def candidate_evaluation_result(self, candidate: Candidate, result_data: dict) -> None:
        """Log individual candidate evaluation result."""
        if self._should_log(logging.INFO):
            average_score = result_data['average_score']
            self.logger.info(f"Comprehensive evaluation: candidate gen={candidate.generation_number} achieves μ={average_score:.3f}")


class SelectionLogger(AsyncLogger):
    """Specialized logger for selection operations."""

    def __init__(self, verbose: bool = False):
        super().__init__("darwin.selection", logging.DEBUG if verbose else logging.INFO)
        self.verbose = verbose

    async def promote(self, survivors: Survivors) -> None:
        """Log promotion operation."""
        if self._should_log(logging.INFO):
            next_generation = survivors.iteration + 1
            self.logger.info(f"Pareto promotion: advancing {len(survivors.candidates)} candidates to generation {next_generation}")

            if self.verbose and self._should_log(logging.DEBUG):
                for candidate in survivors.candidates:
                    score = candidate.average_score()
                    self.logger.debug(f"Candidate gen={candidate.generation_number}: μ={score:.3f}")

    async def pareto_filtering(self, frontier_size: int) -> None:
        """Log Pareto frontier filtering."""
        if self._should_log(logging.DEBUG):
            self.logger.debug(f"Pareto filtering: {frontier_size} candidates in active frontier")

    async def best_candidate_selection(self, candidate: Candidate, pool_size: int) -> None:
        """Log best candidate selection."""
        if self._should_log(logging.INFO):
            score = candidate.average_score()
            self.logger.info(f"Optimal candidate selected: gen={candidate.generation_number}, μ={score:.3f}")

            if self.verbose and self._should_log(logging.DEBUG):
                self.logger.debug(f"Selection pool: {pool_size} candidates")

    async def final_instruction(self, instruction: str) -> None:
        """Log final optimized instruction."""
        if self._should_log(logging.INFO):
            truncated = instruction[:100] + "..." if len(instruction) > 100 else instruction
            self.logger.info(f"Final optimized instruction: {truncated}")


class GenerationLogger(AsyncLogger):
    """Specialized logger for generation operations."""

    def __init__(self, verbose: bool = False):
        super().__init__("darwin.generation", logging.DEBUG if verbose else logging.INFO)
        self.verbose = verbose

    async def generate(self, parents: Parents, newborns: NewBorns) -> None:
        """Log generation operation."""
        if self._should_log(logging.INFO):
            self.logger.info(f"Generated {len(newborns.candidates)} new candidates from {len(parents.candidates)} parents")

    async def mutation_attempt(self, candidate: Candidate, attempt: int, max_attempts: int) -> None:
        """Log mutation attempt."""
        if self._should_log(logging.DEBUG):
            self.logger.debug(f"Mutation attempt {attempt}/{max_attempts} for candidate gen={candidate.generation_number}")

    async def mutation_success(self, original: Candidate, mutated: Candidate) -> None:
        """Log successful mutation."""
        if self._should_log(logging.DEBUG):
            self.logger.debug(f"Mutation successful: gen={original.generation_number} → gen={mutated.generation_number}")

    async def mutation_failure(self, candidate: Candidate, reason: str) -> None:
        """Log mutation failure."""
        if self._should_log(logging.WARNING):
            self.logger.warning(f"Mutation failed for candidate gen={candidate.generation_number}: {reason}")


class StrategyLogger(AsyncLogger):
    """Specialized logger for strategy operations."""

    def __init__(self, verbose: bool = False):
        super().__init__("darwin.strategy", logging.DEBUG if verbose else logging.INFO)
        self.verbose = verbose

    async def start_compilation(self, student_type: str, trainset_size: int, devset_size: int) -> None:
        """Log compilation start."""
        if self._should_log(logging.INFO):
            self.logger.info(f"Starting Darwin optimization: {student_type}, trainset={trainset_size}, devset={devset_size}")

    async def finish_compilation(self, success: bool, generations: int, final_score: float) -> None:
        """Log compilation completion."""
        if self._should_log(logging.INFO):
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"Darwin optimization {status}: {generations} generations, final μ={final_score:.3f}")

    async def start_iteration(self, iteration: int, population_size: int) -> None:
        """Log iteration start."""
        if self._should_log(logging.INFO):
            self.logger.info(f"Starting iteration {iteration} with population size {population_size}")

    async def finish_iteration(self, iteration: int, survivors_count: int) -> None:
        """Log iteration completion."""
        if self._should_log(logging.INFO):
            self.logger.info(f"Iteration {iteration} complete: {survivors_count} survivors")

    async def budget_exhausted(self, consumed: int, max_calls: int) -> None:
        """Log budget exhaustion."""
        if self._should_log(logging.WARNING):
            self.logger.warning(f"Budget exhausted: {consumed}/{max_calls} LM calls consumed")

    async def patience_triggered(self, iterations_without_improvement: int, patience: int) -> None:
        """Log patience mechanism triggering."""
        if self._should_log(logging.INFO):
            self.logger.info(f"Patience triggered: {iterations_without_improvement}/{patience} iterations without improvement")


class LoggerFactory:
    """Factory for creating and configuring async loggers."""

    @staticmethod
    def create_evaluation_logger(verbose: bool = False) -> EvaluationLogger:
        """Create evaluation logger."""
        return EvaluationLogger(verbose=verbose)

    @staticmethod
    def create_selection_logger(verbose: bool = False) -> SelectionLogger:
        """Create selection logger."""
        return SelectionLogger(verbose=verbose)

    @staticmethod
    def create_generation_logger(verbose: bool = False) -> GenerationLogger:
        """Create generation logger."""
        return GenerationLogger(verbose=verbose)

    @staticmethod
    def create_strategy_logger(verbose: bool = False) -> StrategyLogger:
        """Create strategy logger."""
        return StrategyLogger(verbose=verbose)

    @staticmethod
    def create_all_loggers(verbose: bool = False) -> Dict[str, AsyncLogger]:
        """Create all specialized loggers."""
        return {
            'evaluation': LoggerFactory.create_evaluation_logger(verbose),
            'selection': LoggerFactory.create_selection_logger(verbose),
            'generation': LoggerFactory.create_generation_logger(verbose),
            'strategy': LoggerFactory.create_strategy_logger(verbose),
        }
