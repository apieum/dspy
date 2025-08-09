"""GEPA: Genetic-Pareto optimizer for compound AI systems.

A well-structured module implementation of the GEPA optimization algorithm
from "GEPA: Genetic-Pareto Optimization for Task-Specific Instruction Evolution".
"""

# Main GEPA implementation
from .core import GEPA
from .data.candidate import Candidate
from .data.cohort import Cohort

# Protocol interfaces
from .budget import Budget
from .selection import Selector
from .generation import Generator
from .evaluation import Evaluator

# Business step implementations
from .budget import LMCallsBudget, IterationBudget, AdaptiveBudget
from .selection import ParetoFrontier
from .generation import ReflectivePromptMutation, SystemAwareMerge
from .evaluation import GEPAEvaluator, FullTaskScores, ParentFastCompare
from .evaluation.trace_collector import EnhancedTraceCollector
from .evaluation.feedback import FeedbackResult, EvaluationTrace, ModuleFeedback

# Factory functions are now static methods on GEPA class

__all__ = [
    # Core classes
    'GEPA',
    'Candidate',
    'Cohort',

    # Protocol interfaces
    'Budget',
    'Selector',
    'Generator',
    'Evaluator',

    # Business step implementations
    'LMCallsBudget',
    'IterationBudget',
    'AdaptiveBudget',
    'ParetoFrontier',
    'ReflectivePromptMutation',
    'SystemAwareMerge',
    'GEPAEvaluator',
    'FullTaskScores',
    'ParentFastCompare',
    'EnhancedTraceCollector',
    'FeedbackResult',
    'EvaluationTrace',
    'ModuleFeedback',
]
