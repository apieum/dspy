"""GEPA: Genetic-Pareto optimizer for compound AI systems.

A well-structured module implementation of the GEPA optimization algorithm
from "GEPA: Genetic-Pareto Optimization for Task-Specific Instruction Evolution".
"""

# Budget system
from .budget import BudgetTracker, CombinedBudget, IterationBudget, LLMCallsBudget

# Data structures and candidate management
from .data import (
    CandidateLineage,
    CandidatePool,
    EvaluationTrace,
    FeedbackResult,
    ModuleFeedback,
    ScoreMatrix,
    SplitDataset,
    TrainingDataset,
)

# Feedback collection and mutation
from .feedback import (
    EnhancedFeedbackCollector,
    FeedbackCollector,
    ModuleSelector,
    PromptMutator,
    ReflectivePromptMutator,
    RoundRobinModuleSelector,
)

# Candidate generation strategies
from .generators import (
    CandidateGenerator,
    CompositeGenerator,
    CrossoverGenerator,
    MutationGenerator,
)

# Selection strategies
from .selection import CandidateSelector, ParetoCandidateSelector

# Main GEPA class from modular structure
from .core import GEPA

__all__ = [
    # Main class
    'GEPA',
    
    # Budget system
    'BudgetTracker',
    'LLMCallsBudget',
    'IterationBudget', 
    'CombinedBudget',
    
    # Data structures
    'TrainingDataset',
    'SplitDataset',
    'CandidatePool',
    'CandidateLineage',
    'ScoreMatrix',
    'FeedbackResult',
    'EvaluationTrace',
    'ModuleFeedback',
    
    # Generators
    'CandidateGenerator',
    'MutationGenerator',
    'CrossoverGenerator',
    'CompositeGenerator',
    
    # Selection
    'CandidateSelector',
    'ParetoCandidateSelector',
    
    # Feedback
    'FeedbackCollector',
    'EnhancedFeedbackCollector',
    'PromptMutator',
    'ReflectivePromptMutator',
    'ModuleSelector',
    'RoundRobinModuleSelector',
]