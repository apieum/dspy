"""Concrete implementations of GEPA optimization strategies.

This module provides working implementations of each strategy protocol,
allowing users to compose complete GEPA optimizers by mixing and matching
different strategies based on their research needs.
"""

import random
from typing import List, Optional, Callable, Any

import dspy
from dspy import Module

from .protocols import (
    Budget,
    Scoring, 
    Selection,
    Generator,
    Evaluator
)
from ..data.candidate import Candidate
from ..data.score_matrix import ScoreMatrix
from ..data.candidate_pool import CandidatePool
from ..generation.generation import Generation


class LLMCallsBudgetStrategy(Budget):
    """Budget strategy that tracks LLM API calls."""
    
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.consumed_calls = 0
        
    def can_afford(self, cost: int, operation_type: str) -> bool:
        return self.consumed_calls + cost <= self.max_calls
        
    def consume(self, cost: int, operation_type: str) -> None:
        self.consumed_calls += cost
        
    def is_empty(self) -> bool:
        return self.consumed_calls >= self.max_calls
        
    def get_remaining(self) -> int:
        return max(0, self.max_calls - self.consumed_calls)


class IterationBudgetStrategy(Budget):
    """Budget strategy that limits by number of iterations."""
    
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
    def can_afford(self, cost: int, operation_type: str) -> bool:
        return self.current_iteration < self.max_iterations
        
    def consume(self, cost: int, operation_type: str) -> None:
        if operation_type == "iteration":
            self.current_iteration += 1
            
    def is_empty(self) -> bool:
        return self.current_iteration >= self.max_iterations
        
    def get_remaining(self) -> int:
        return max(0, self.max_iterations - self.current_iteration)


class ParetoScoringStrategy(Scoring):
    """Scoring strategy using Pareto-frontier evaluation."""
    
    def __init__(self, metric: Callable):
        self.metric = metric
        
    def calculate_scores(self, candidates: List[Candidate], 
                        data: List[dspy.Example], 
                        candidate_pool: CandidatePool) -> ScoreMatrix:
        """Evaluate candidates on tasks and add them to the candidate pool."""
        
        for candidate in candidates:
            # Evaluation phase: score candidate on each task (minibatch)
            total_score = 0.0
            
            for task_id, example in enumerate(data):
                score = candidate.evaluate_on_example(example, self.metric)
                # Store score in candidate during evaluation
                candidate.set_task_score(task_id, score)
                total_score += score
                
            # Store average as overall fitness
            if data:
                avg_score = total_score / len(data)
                candidate.add_fitness_score(avg_score)
            
            # Add candidate to pool - pool reads candidate's scores and updates matrix
            candidate_pool.add_candidate(candidate)
        
        # Return the candidate pool's score matrix
        return candidate_pool.score_matrix
        
    def get_metric(self) -> Callable:
        return self.metric


class ElitistFilteringStrategy(Selection):
    """Filtering strategy that keeps top N performers."""
    
    def __init__(self, keep_top_n: int = 5):
        self.keep_top_n = keep_top_n
        
    def filter(self, candidate_pool: CandidatePool, score_matrix: ScoreMatrix) -> List[Candidate]:
        # Use filtering strategy to get top performing candidates by balanced scores
        from .filtering_strategies import BalancedTopStrategy
        
        strategy = BalancedTopStrategy(keep_top_n=self.keep_top_n)
        return candidate_pool.filter_top(candidate_pool.size(), strategy)


class MutationGenerationStrategy(Generator):
    """Generation strategy using mutation of existing candidates."""
    
    def __init__(self, mutation_rate: float = 0.3, population_size: int = 10):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        
    def generate(self, parent_candidates: List[Candidate], 
                feedback_data: List[dspy.Example], 
                iteration: int) -> Generation:
        
        new_candidates = []
        
        for i in range(self.population_size):
            if parent_candidates and random.random() < self.mutation_rate:
                # Select random parent for mutation
                parent = random.choice(parent_candidates)
                
                # Create mutated copy (simplified - real implementation would use proper mutation)
                mutated_module = parent.module.deepcopy()
                
                new_candidate = Candidate(
                    module=mutated_module,
                    parent_ids=[parent.candidate_id],
                    generation_number=iteration
                )
                new_candidates.append(new_candidate)
                
        return Generation(
            candidates=new_candidates,
            generation_id=iteration,
            iteration=iteration
        )


class CrossoverGenerationStrategy(Generator):
    """Generation strategy using crossover between candidates."""
    
    def __init__(self, crossover_rate: float = 0.7, population_size: int = 10):
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        
    def generate(self, parent_candidates: List[Candidate],
                feedback_data: List[dspy.Example],
                iteration: int) -> Generation:
        
        new_candidates = []
        
        for i in range(self.population_size):
            if len(parent_candidates) >= 2 and random.random() < self.crossover_rate:
                # Select two parents for crossover
                parent1 = random.choice(parent_candidates)
                parent2 = random.choice(parent_candidates)
                
                # Create crossover child (simplified - real implementation would merge modules)
                child_module = parent1.module.deepcopy()  # Simplified crossover
                
                new_candidate = Candidate(
                    module=child_module,
                    parent_ids=[parent1.candidate_id, parent2.candidate_id],
                    generation_number=iteration
                )
                new_candidates.append(new_candidate)
                
        return Generation(
            candidates=new_candidates,
            generation_id=iteration,
            iteration=iteration
        )


class PromotionEvaluationStrategy(Evaluator):
    """Evaluation strategy that promotes promising candidates."""
    
    def __init__(self, metric: Callable, promotion_threshold: float = 0.5):
        self.metric = metric
        self.promotion_threshold = promotion_threshold
        
    def evaluate(self, generation: Generation, 
                evaluation_data: List[dspy.Example]) -> Generation:
        
        promoted_candidates = []
        
        for candidate in generation.candidates:
            # Evaluate candidate performance
            score = candidate.evaluate_on_batch(evaluation_data, self.metric)
            
            # Promote if above threshold
            if score >= self.promotion_threshold:
                promoted_candidates.append(candidate)
                
        return Generation(
            candidates=promoted_candidates,
            generation_id=generation.generation_id,
            iteration=generation.iteration
        )
        
    def get_metric(self) -> Callable:
        return self.metric


class DiversityFilteringStrategy(Selection):
    """Filtering strategy that maintains population diversity."""
    
    def __init__(self, diversity_weight: float = 0.3, keep_top_n: int = 5):
        self.diversity_weight = diversity_weight
        self.keep_top_n = keep_top_n
        
    def filter(self, candidate_pool: CandidatePool, score_matrix: ScoreMatrix) -> List[Candidate]:
        # Use filtering strategies for diversity-aware filtering
        from .filtering_strategies import ParetoFrontierStrategy, BalancedTopStrategy
        
        # First get Pareto frontier candidates
        pareto_strategy = ParetoFrontierStrategy()
        pareto_candidates = candidate_pool.filter_best(pareto_strategy)
        
        # If we need more candidates, supplement with balanced top performers
        if len(pareto_candidates) < self.keep_top_n:
            balanced_strategy = BalancedTopStrategy(keep_top_n=self.keep_top_n)
            balanced_candidates = candidate_pool.filter_top(candidate_pool.size(), balanced_strategy)
            
            # Add candidates not already in Pareto set
            pareto_ids = {c.candidate_id for c in pareto_candidates}
            for candidate in balanced_candidates:
                if candidate.candidate_id not in pareto_ids and len(pareto_candidates) < self.keep_top_n:
                    pareto_candidates.append(candidate)
        
        return pareto_candidates[:self.keep_top_n]


class AdaptiveBudgetStrategy(Budget):
    """Budget strategy that adapts allocation based on progress."""
    
    def __init__(self, total_budget: int, adaptation_factor: float = 1.2):
        self.total_budget = total_budget
        self.consumed_budget = 0
        self.adaptation_factor = adaptation_factor
        self.recent_improvements = []
        
    def can_afford(self, cost: int, operation_type: str) -> bool:
        return self.consumed_budget + cost <= self.total_budget
        
    def consume(self, cost: int, operation_type: str) -> None:
        self.consumed_budget += cost
        
    def is_empty(self) -> bool:
        return self.consumed_budget >= self.total_budget
        
    def get_remaining(self) -> int:
        return max(0, self.total_budget - self.consumed_budget)
        
    def record_improvement(self, improvement: float) -> None:
        """Record performance improvement for adaptive allocation."""
        self.recent_improvements.append(improvement)
        # Keep only recent history
        if len(self.recent_improvements) > 10:
            self.recent_improvements = self.recent_improvements[-10:]