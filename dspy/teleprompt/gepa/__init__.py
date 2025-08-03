"""GEPA: Genetic-Pareto optimizer for compound AI systems.

A well-structured module implementation of the GEPA optimization algorithm
from "GEPA: Genetic-Pareto Optimization for Task-Specific Instruction Evolution".
"""

# Main GEPA implementation
from .core import GEPA
from .data.candidate import Candidate
from .data.score_matrix import ScoreMatrix
from .data.candidate_pool import CandidatePool
from .data.cohort import Cohort, FilteredCohort

# Protocol interfaces
from .budget import Budget
from .scoring import Scoring
from .selection import Selection 
from .generation import Generator
from .evaluation import Evaluator
from .filtering import Filtering

# Business step implementations
from .budget import LLMCallsBudget, IterationBudget, AdaptiveBudget
from .scoring import ParetoScoring
from .selection import ParetoSelection
from .generation import MutationGenerator, CrossoverGenerator
from .generation.reflective_mutation import ReflectiveMutation
from .evaluation import PromotionEvaluator
from .evaluation.trace_collector import EnhancedTraceCollector
from .evaluation.feedback import FeedbackResult, EvaluationTrace, ModuleFeedback
from .filtering import ParetoFrontier, TopScores, BalancedTop, Threshold, Diversity

# Factory functions are now static methods on GEPA class

__all__ = [
    # Core classes
    'GEPA',
    'Candidate', 
    'Cohort',
    'FilteredCohort',
    'ScoreMatrix',
    'CandidatePool',
    
    # Protocol interfaces
    'Budget',
    'Scoring', 
    'Selection',
    'Generator',
    'Evaluator',
    'Filtering',
    
    # Business step implementations
    'LLMCallsBudget',
    'IterationBudget',
    'AdaptiveBudget',
    'ParetoScoring',
    'ParetoSelection',
    'MutationGenerator',
    'CrossoverGenerator',
    'ReflectiveMutation',
    'PromotionEvaluator',
    'EnhancedTraceCollector',
    'FeedbackResult',
    'EvaluationTrace',
    'ModuleFeedback',
    
    # Filtering implementations
    'ParetoFrontier',
    'TopScores',
    'BalancedTop',
    'Threshold',
    'Diversity',
]