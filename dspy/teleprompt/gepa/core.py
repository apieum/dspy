"""GEPA Framework - Strategy-based prompt optimization framework.

This module implements GEPA as a clean, configurable framework using
dependency injection and the strategy pattern for research flexibility.
"""

import logging
from typing import List, Optional

import dspy
from dspy import Module

from .strategies.protocols import (
    Budget,
    Scoring, 
    Selection,
    Generator,
    Evaluator
)
from .data.candidate import Candidate
from .data.candidate_pool import CandidatePool
from .generation.generation import Generation

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
        G1 = Generation([initial_candidate], generation_id=0, iteration=0)
        
        logger.info("Initialized with first generation")
        
        # Step 2: Recursive optimization  
        result = self.next_generation(G1, pareto_data, feedback_data)
        
        logger.info("GEPA framework optimization complete")
        return result
    
    def next_generation(self, generation: Generation,
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
            generation: Current generation to process
            pareto_data: Data for scoring strategy
            feedback_data: Data for generation strategy
            
        Returns:
            Best optimized module when termination criteria met
        """
        self.current_iteration += 1
        logger.info(f"Processing generation {generation.generation_id}, iteration {self.current_iteration}")
        
        # Termination conditions
        if self.budget.is_empty():
            logger.info("Budget exhausted - terminating optimization")
            return self._show_results(pareto_data)
            
        if generation.is_empty():
            logger.info("Empty generation - terminating optimization")
            return self._show_results(pareto_data)
        
        # Phase 1: Calculate scores and add candidates to pool
        logger.debug(f"Scoring {generation.size()} candidates")
        
        # Consume budget for scoring
        scoring_cost = len(generation.candidates) * len(pareto_data)
        if not self.budget.can_afford(scoring_cost, "scoring"):
            logger.warning("Insufficient budget for scoring - terminating")
            return self._show_results(pareto_data)
        
        # Scoring strategy calculates scores and adds candidates to pool
        score_matrix = self.scoring.calculate_scores(generation.candidates, pareto_data, self.candidate_pool)
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
        
        new_generation = self.generator.generate(filtered_candidates, feedback_data, self.current_iteration)
        self.budget.consume(generation_cost, "generation")
        
        # Phase 4: Evaluate and filter new generation  
        logger.debug(f"Evaluating {new_generation.size()} new candidates")
        evaluation_cost = len(new_generation.candidates) * len(feedback_data) // 5  # Estimate
        if not self.budget.can_afford(evaluation_cost, "evaluation"):
            logger.warning("Insufficient budget for evaluation - using all candidates")
            filtered_generation = new_generation
        else:
            filtered_generation = self.evaluator.evaluate(new_generation, feedback_data)
            self.budget.consume(evaluation_cost, "evaluation")
        
        # Phase 5: Prepare next generation and recurse
        filtered_generation.generation_id = generation.generation_id + 1
        filtered_generation.iteration = self.current_iteration
        
        logger.debug(f"Recursing with {filtered_generation.size()} promoted candidates")
        return self.next_generation(filtered_generation, pareto_data, feedback_data)
    
    def _show_results(self, evaluation_data: List[dspy.Example]) -> Module:
        """Select and return the best candidate when optimization terminates."""
        logger.info("Selecting best candidate from optimization results")
        
        # Use balanced top strategy to get the best candidate
        from .strategies.filtering_strategies import BalancedTopStrategy
        strategy = BalancedTopStrategy(keep_top_n=1)
        best_candidates = self.candidate_pool.filter_top(self.candidate_pool.size(), strategy)
        
        if not best_candidates:
            raise RuntimeError("No candidates found in pool - optimization failed")
        
        best_candidate = best_candidates[0]
        best_score = best_candidate.get_average_task_score()
        
        logger.info(f"Selected best candidate (ID: {best_candidate.candidate_id}) with score: {best_score:.4f}")
        logger.info(f"Final pool size: {self.candidate_pool.size()} candidates across {self.candidate_pool.generation_count()} generations")
        
        compiled_module = best_candidate.module
        compiled_module._compiled = True
        return compiled_module
    
