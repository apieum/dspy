"""GEPA: Genetic-Pareto optimizer for compound AI systems.

A well-structured module implementation of the GEPA optimization algorithm
from "GEPA: Genetic-Pareto Optimization for Task-Specific Instruction Evolution".
"""

# Main GEPA implementation
from .core import GEPA
from .data.candidate import Candidate
from .data.candidate_pool import CandidatePool
from .data.cohort import Cohort

# Protocol interfaces
from .budget import Budget
from .selection import Selection
from .generation import Generator
from .evaluation import Evaluator

# Business step implementations
from .budget import LLMCallsBudget, IterationBudget, AdaptiveBudget
from .selection import ParetoSelection
from .generation import MutationGenerator, CrossoverGenerator
from .generation.reflective_mutation import ReflectiveMutation
from .evaluation import PromotionEvaluator
from .evaluation.trace_collector import EnhancedTraceCollector
from .evaluation.feedback import FeedbackResult, EvaluationTrace, ModuleFeedback

# Factory functions are now static methods on GEPA class

__all__ = [
    # Core classes
    'GEPA',
    'Candidate',
    'Cohort',
    'CandidatePool',

    # Protocol interfaces
    'Budget',
    'Selection',
    'Generator',
    'Evaluator',

    # Business step implementations
    'LLMCallsBudget',
    'IterationBudget',
    'AdaptiveBudget',
    'ParetoSelection',
    'MutationGenerator',
    'CrossoverGenerator',
    'ReflectiveMutation',
    'PromotionEvaluator',
    'EnhancedTraceCollector',
    'FeedbackResult',
    'EvaluationTrace',
    'ModuleFeedback',
]
