"""GEPA Framework - Strategy-based prompt optimization framework.

This module implements GEPA as a clean, configurable framework using
dependency injection and the strategy pattern for research flexibility.
"""

import logging
from typing import List, Optional

import dspy
from dspy import Module

from .budget import Budget
from .scoring import Scoring
from .selection import Selection
from .generation import Generator
from .evaluation import Evaluator
from .data.candidate import Candidate
from .data.candidate_pool import CandidatePool
from .data.cohort import Cohort, FilteredCohort

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
                 scoring: Scoring,
                 selection: Selection,
                 generator: Generator,
                 evaluator: Evaluator):
        """Initialize GEPA optimization with algorithms for each step.

        Args:
            budget: Manages optimization budget (LLM calls, iterations)
            scoring: Calculates candidate performance (owns metric)
            selection: Selects surviving candidates based on scores
            generator: Creates new candidates via genetic operations
            evaluator: Promotes worthy new candidates (owns metric)
        """
        self.budget = budget
        self.scoring = scoring
        self.selection = selection
        self.generator = generator
        self.evaluator = evaluator

        # Framework state
        self.candidate_pool = CandidatePool()
        self.current_iteration = 0

    def compile(self, student: Module,
               pareto_data: List[dspy.Example],
               feedback_data: List[dspy.Example]) -> Module:
        """Main compilation method - entry point for optimization.

        Args:
            student: Initial program to optimize
            pareto_data: Data for performance scoring
            feedback_data: Data for generating feedback/mutations

        Returns:
            Optimized program module
        """
        logger.info("Starting GEPA framework optimization...")

        # Step 1: Initialize first candidate generation
        initial_candidate = Candidate(student.deepcopy(), generation_number=0)
        gen_0 = Cohort([initial_candidate])

        logger.info("Initialized with first generation")

        # Step 2: Recursive optimization
        result = self.next_generation(gen_0, pareto_data, feedback_data)

        logger.info("GEPA framework optimization complete")
        return result

    def next_generation(self, cohort: Cohort,
                       pareto_data: List[dspy.Example],
                       feedback_data: List[dspy.Example]) -> Module:
        """Recursive optimization step implementing the core algorithm.

        This method expresses the GEPA algorithm structure clearly:
        1. Check termination conditions
        2. Score candidates on performance data
        3. Filter candidates based on scores
        4. Generate new candidates from survivors
        5. Evaluate and promote new candidates
        6. Recurse to next generation

        Args:
            cohort: Current cohort to process
            pareto_data: Data for scoring strategy
            feedback_data: Data for generation strategy

        Returns:
            Best optimized module when termination criteria met
        """
        self.current_iteration += 1
        logger.info(f"Processing cohort {cohort.iteration_id}, iteration {self.current_iteration}")

        # Termination conditions
        if self.budget.is_empty():
            logger.info("Budget exhausted - terminating optimization")
            return self._show_results(pareto_data)

        if cohort.is_empty():
            logger.info("Empty generation - terminating optimization")
            return self._show_results(pareto_data)

        # Phase 1: Calculate scores and add candidates to pool
        logger.debug(f"Scoring {cohort.size()} candidates")

        # Consume budget for scoring
        scoring_cost = len(cohort.candidates) * len(pareto_data)
        if not self.budget.can_afford(scoring_cost, "scoring"):
            logger.warning("Insufficient budget for scoring - terminating")
            return self._show_results(pareto_data)

        # Scoring strategy calculates scores and adds candidates to pool
        score_matrix = self.scoring.calculate_scores(cohort.candidates, pareto_data, self.candidate_pool)
        self.budget.consume(scoring_cost, "scoring")

        # Phase 2: Filter candidates based on performance
        logger.debug("Filtering candidates based on scores")
        filtered_candidates = self.selection.filter(self.candidate_pool, score_matrix)

        # Phase 3: Generate new candidates from filtered survivors
        logger.debug("Generating new candidates")
        generation_cost = len(filtered_candidates) * len(feedback_data) // 10  # Estimate
        if not self.budget.can_afford(generation_cost, "generation"):
            logger.warning("Insufficient budget for generation - terminating")
            return self._show_results(pareto_data)

        new_cohort = self.generator.generate(filtered_candidates, feedback_data, self.current_iteration)
        self.budget.consume(generation_cost, "generation")

        # Phase 4: Evaluate and filter new generation
        logger.debug(f"Evaluating {new_cohort.size()} new candidates")
        evaluation_cost = len(new_cohort.candidates) * len(feedback_data) // 5  # Estimate
        if not self.budget.can_afford(evaluation_cost, "evaluation"):
            logger.warning("Insufficient budget for evaluation - using all candidates")
            filtered_cohort = new_cohort
        else:
            filtered_cohort = self.evaluator.evaluate(new_cohort, feedback_data)
            self.budget.consume(evaluation_cost, "evaluation")

        # Phase 5: Prepare next cohort and recurse
        logger.debug(f"Recursing with {filtered_cohort.size()} promoted candidates")
        return self.next_generation(filtered_cohort, pareto_data, feedback_data)

    def _show_results(self, evaluation_data: List[dspy.Example]) -> Module:
        """Select and return the best candidate when optimization terminates."""
        logger.info("Selecting best candidate from optimization results")

        # Get the best candidate per task using a cohort accumulator
        from .data import Cohort
        top_candidates_cohort = Cohort([])
        self.candidate_pool.filter_top(top_candidates_cohort)

        if not top_candidates_cohort.candidates:
            raise RuntimeError("No candidates found in pool - optimization failed")

        # Select the candidate with the highest average score from task winners
        best_candidate = max(top_candidates_cohort.candidates, key=lambda c: c.average_task_score())
        best_score = best_candidate.average_task_score()

        logger.info(f"Selected best candidate with score: {best_score:.4f}")
        logger.info(f"Final pool size: {self.candidate_pool.size()} candidates across {self.candidate_pool.generation_count()} generations")

        compiled_module = best_candidate.module
        compiled_module._compiled = True
        return compiled_module

    @staticmethod
    def create_basic(metric, max_calls: int = 1000, population_size: int = 10) -> "GEPA":
        """Create a basic GEPA optimizer with standard settings.
        
        Uses:
        - LLM calls budget tracking
        - Pareto scoring
        - Elitist selection (keep top 5)
        - Mutation-based generation
        - Promotion-based evaluation
        
        Args:
            metric: Evaluation metric for candidate scoring
            max_calls: Maximum LLM API calls budget
            population_size: Number of candidates per generation
            
        Returns:
            Configured GEPA optimizer ready for compilation
        """
        from .budget import LLMCallsBudget
        from .scoring import ParetoScoring
        from .selection import ParetoSelection
        from .generation import MutationGenerator
        from .evaluation import PromotionEvaluator
        
        return GEPA(
            budget=LLMCallsBudget(max_calls),
            scoring=ParetoScoring(metric),
            selection=ParetoSelection(),
            generator=MutationGenerator(
                mutation_rate=0.3, 
                population_size=population_size
            ),
            evaluator=PromotionEvaluator(
                metric=metric,
                promotion_threshold=0.5
            )
        )


    @staticmethod
    def create_iteration_limited(metric, max_iterations: int = 20, population_size: int = 8) -> "GEPA":
        """Create a GEPA optimizer limited by iterations rather than LLM calls.
        
        Useful for research scenarios where you want consistent iteration counts
        regardless of population size or evaluation complexity.
        
        Args:
            metric: Evaluation metric for candidate scoring
            max_iterations: Maximum number of optimization iterations
            population_size: Number of candidates per generation
            
        Returns:
            Configured GEPA optimizer with iteration-based budget
        """
        from .budget import IterationBudget
        from .scoring import ParetoScoring
        from .selection import ParetoSelection
        from .generation import MutationGenerator
        from .evaluation import PromotionEvaluator
        
        return GEPA(
            budget=IterationBudget(max_iterations),
            scoring=ParetoScoring(metric),
            selection=ParetoSelection(),
            generator=MutationGenerator(
                mutation_rate=0.4,
                population_size=population_size
            ),
            evaluator=PromotionEvaluator(
                metric=metric,
                promotion_threshold=0.6
            )
        )

    @staticmethod
    def create_adaptive(metric, total_budget: int = 2000, population_size: int = 12) -> "GEPA":
        """Create an adaptive GEPA optimizer that adjusts based on progress.
        
        Uses adaptive budget allocation that responds to performance improvements.
        Good for long-running optimizations where you want intelligent resource use.
        
        Args:
            metric: Evaluation metric for candidate scoring
            total_budget: Total budget for adaptive allocation
            population_size: Number of candidates per generation
            
        Returns:
            Configured GEPA optimizer with adaptive strategies
        """
        from .budget import AdaptiveBudget
        from .scoring import ParetoScoring
        from .selection import DiversitySelection
        from .generation import CrossoverGenerator
        from .evaluation import PromotionEvaluator
        
        return GEPA(
            budget=AdaptiveBudget(
                total_budget=total_budget,
                adaptation_factor=1.3
            ),
            scoring=ParetoScoring(metric),
            selection=DiversitySelection(
                diversity_weight=0.35,
                keep_top_n=6
            ),
            generator=CrossoverGenerator(
                crossover_rate=0.6,
                population_size=population_size
            ),
            evaluator=PromotionEvaluator(
                metric=metric,
                promotion_threshold=0.45
            )
        )

    @staticmethod
    def create_research(metric, max_calls: int = 3000, population_size: int = 20) -> "GEPA":
        """Create a GEPA optimizer configured for research scenarios.
        
        Uses larger populations and budgets for comprehensive exploration.
        Balanced between exploitation and exploration.
        
        Args:
            metric: Evaluation metric for candidate scoring
            max_calls: Maximum LLM API calls budget
            population_size: Number of candidates per generation
            
        Returns:
            Configured GEPA optimizer for research use
        """
        from .budget import LLMCallsBudget
        from .scoring import ParetoScoring
        from .selection import DiversitySelection
        from .generation import CrossoverGenerator
        from .evaluation import PromotionEvaluator
        
        return GEPA(
            budget=LLMCallsBudget(max_calls),
            scoring=ParetoScoring(metric),
            selection=DiversitySelection(
                diversity_weight=0.3,
                keep_top_n=8
            ),
            generator=CrossoverGenerator(
                crossover_rate=0.7,
                population_size=population_size
            ),
            evaluator=PromotionEvaluator(
                metric=metric,
                promotion_threshold=0.3  # More permissive for exploration
            )
        )
