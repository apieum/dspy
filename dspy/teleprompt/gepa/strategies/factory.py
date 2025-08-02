"""Factory functions for creating common GEPA optimizer configurations.

This module provides convenient factory functions for assembling complete
GEPA optimizers with sensible defaults for different research scenarios.
"""

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import GEPA
from .implementations import (
    LLMCallsBudgetStrategy,
    IterationBudgetStrategy,
    ParetoScoringStrategy,
    ElitistFilteringStrategy,
    DiversityFilteringStrategy,
    MutationGenerationStrategy,
    CrossoverGenerationStrategy,
    PromotionEvaluationStrategy,
    AdaptiveBudgetStrategy
)


def create_basic_gepa(metric: Callable, 
                     max_calls: int = 1000,
                     population_size: int = 10) -> "GEPA":
    """Create a basic GEPA optimizer with standard settings.
    
    Uses:
    - LLM calls budget tracking
    - Pareto scoring
    - Elitist filtering (keep top 5)
    - Mutation-based generation
    - Promotion-based evaluation
    
    Args:
        metric: Evaluation metric for candidate scoring
        max_calls: Maximum LLM API calls budget
        population_size: Number of candidates per generation
        
    Returns:
        Configured GEPA optimizer ready for compilation
    """
    # Import here to avoid circular imports
    from ..core import GEPA
    
    return GEPA(
        budget=LLMCallsBudgetStrategy(max_calls),
        scoring=ParetoScoringStrategy(metric),
        selection=ElitistFilteringStrategy(keep_top_n=5),
        generator=MutationGenerationStrategy(
            mutation_rate=0.3, 
            population_size=population_size
        ),
        evaluator=PromotionEvaluationStrategy(
            metric=metric,
            promotion_threshold=0.5
        )
    )


def create_diversity_gepa(metric: Callable,
                         max_calls: int = 1500,
                         population_size: int = 15) -> "GEPA":
    """Create a diversity-focused GEPA optimizer.
    
    Uses:
    - LLM calls budget tracking
    - Pareto scoring
    - Diversity-aware filtering
    - Crossover-based generation for variety
    - Promotion-based evaluation
    
    Args:
        metric: Evaluation metric for candidate scoring
        max_calls: Maximum LLM API calls budget
        population_size: Number of candidates per generation
        
    Returns:
        Configured GEPA optimizer optimized for diversity
    """
    # Import here to avoid circular imports
    from ..core import GEPA
    
    return GEPA(
        budget=LLMCallsBudgetStrategy(max_calls),
        scoring=ParetoScoringStrategy(metric),
        selection=DiversityFilteringStrategy(
            diversity_weight=0.4,
            keep_top_n=7
        ),
        generator=CrossoverGenerationStrategy(
            crossover_rate=0.8,
            population_size=population_size
        ),
        evaluator=PromotionEvaluationStrategy(
            metric=metric,
            promotion_threshold=0.4  # Lower threshold for more diversity
        )
    )


def create_iteration_limited_gepa(metric: Callable,
                                 max_iterations: int = 20,
                                 population_size: int = 8) -> "GEPA":
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
    # Import here to avoid circular imports
    from ..core import GEPA
    
    return GEPA(
        budget=IterationBudgetStrategy(max_iterations),
        scoring=ParetoScoringStrategy(metric),
        selection=ElitistFilteringStrategy(keep_top_n=4),
        generator=MutationGenerationStrategy(
            mutation_rate=0.4,
            population_size=population_size
        ),
        evaluator=PromotionEvaluationStrategy(
            metric=metric,
            promotion_threshold=0.6
        )
    )


def create_adaptive_gepa(metric: Callable,
                        total_budget: int = 2000,
                        population_size: int = 12) -> "GEPA":
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
    # Import here to avoid circular imports
    from ..core import GEPA
    
    return GEPA(
        budget=AdaptiveBudgetStrategy(
            total_budget=total_budget,
            adaptation_factor=1.3
        ),
        scoring=ParetoScoringStrategy(metric),
        selection=DiversityFilteringStrategy(
            diversity_weight=0.35,
            keep_top_n=6
        ),
        generator=CrossoverGenerationStrategy(
            crossover_rate=0.6,
            population_size=population_size
        ),
        evaluator=PromotionEvaluationStrategy(
            metric=metric,
            promotion_threshold=0.45
        )
    )


def create_research_gepa(metric: Callable,
                        max_calls: int = 3000,
                        population_size: int = 20) -> "GEPA":
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
    # Import here to avoid circular imports
    from ..core import GEPA
    
    return GEPA(
        budget=LLMCallsBudgetStrategy(max_calls),
        scoring=ParetoScoringStrategy(metric),
        selection=DiversityFilteringStrategy(
            diversity_weight=0.3,
            keep_top_n=8
        ),
        generator=CrossoverGenerationStrategy(
            crossover_rate=0.7,
            population_size=population_size
        ),
        evaluator=PromotionEvaluationStrategy(
            metric=metric,
            promotion_threshold=0.3  # More permissive for exploration
        )
    )