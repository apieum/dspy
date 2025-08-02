"""Optimization components for GEPA.

This module defines the protocol interfaces and concrete implementations
for each step of the GEPA optimization process.
"""

# Protocol interfaces
from .protocols import (
    Budget,
    Scoring,
    Selection,
    Generator,
    Evaluator
)

# Concrete implementations
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

# Factory functions
from .factory import (
    create_basic_gepa,
    create_diversity_gepa,
    create_iteration_limited_gepa,
    create_adaptive_gepa,
    create_research_gepa
)

# Filtering strategies
from .filtering_strategies import (
    ParetoFrontierStrategy,
    TopScoresStrategy,
    BalancedTopStrategy,
    ThresholdStrategy,
    DiversityStrategy
)

__all__ = [
    # Protocol interfaces
    'Budget',
    'Scoring', 
    'Selection',
    'Generator',
    'Evaluator',
    
    # Concrete implementations
    'LLMCallsBudgetStrategy',
    'IterationBudgetStrategy', 
    'ParetoScoringStrategy',
    'ElitistFilteringStrategy',
    'DiversityFilteringStrategy',
    'MutationGenerationStrategy',
    'CrossoverGenerationStrategy',
    'PromotionEvaluationStrategy',
    'AdaptiveBudgetStrategy',
    
    # Factory functions
    'create_basic_gepa',
    'create_diversity_gepa',
    'create_iteration_limited_gepa', 
    'create_adaptive_gepa',
    'create_research_gepa',
    
    # Filtering strategies  
    'ParetoFrontierStrategy',
    'TopScoresStrategy',
    'BalancedTopStrategy',
    'ThresholdStrategy',
    'DiversityStrategy'
]