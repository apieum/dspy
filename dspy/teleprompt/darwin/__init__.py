"""Darwin - Extensible evolutionary optimization for language model programs.

Darwin is a general-purpose toolkit for building evolutionary optimizers,
with GEPA as the first "recipe" using the framework.
"""

# Main Darwin implementation
from .optimizer import Darwin
from .data.candidate import Candidate
from .data.cohort import Cohort

# Protocol interfaces
from .budget import Budget
from .selection import Selector
from .generation import Generator
from .evaluation import Evaluator

# Business step implementations
from .budget import LMCallsBudget, IterationBudget, AdaptiveBudget
from .selection.pareto import ParetoFrontier
from .generation.mutation import ReflectivePromptMutation
from .generation.system_aware_merge import SystemAwareMerge
from .generation.feedback import FeedbackProvider
from .evaluation.gepa_evaluator import FullTaskScores, ParentFastCompare, GEPATwoPhasesEval
from .evaluation.trace_collector import EnhancedTraceCollector
from .evaluation.feedback import FeedbackResult, EvaluationTrace, ModuleFeedback

# Visualization tools
from .visualization.candidate_tree import CandidateTreeVisualizer

# Result classes and observers
from .result import Result, Success, Failure
from .observers import (
    OptimizerObserver, SelectorObserver, GeneratorObserver, EvaluatorObserver, CandidateObserver,
    ChannelContext, Channel
)

# Async logging system
from .logging import (
    AsyncLogger, EvaluationLogger, SelectionLogger, GenerationLogger,
    StrategyLogger, LoggerFactory
)

# Strategy and configuration
from .strategy import BaseStrategy, GEPAStrategy
from .config import DarwinConfig

# GEPA convenience optimizers
from .gepa_optimizers import GEPAMute, GEPAAdaptive

__all__ = [
    # Core classes
    'Darwin',
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
    'FeedbackProvider',
    'FullTaskScores',
    'ParentFastCompare',
    'GEPATwoPhasesEval',
    'EnhancedTraceCollector',
    'FeedbackResult',
    'EvaluationTrace',
    'ModuleFeedback',

    # Visualization tools
    'CandidateTreeVisualizer',

    # Result classes and observers
    'Result', 'Success', 'Failure',
    'OptimizerObserver', 'SelectorObserver', 'GeneratorObserver', 'EvaluatorObserver', 'CandidateObserver',
    'ChannelContext', 'Channel',

    # Async logging system
    'AsyncLogger', 'EvaluationLogger', 'SelectionLogger', 'GenerationLogger',
    'StrategyLogger', 'LoggerFactory',

    # Strategy and configuration
    'BaseStrategy',
    'GEPAStrategy',
    'DarwinConfig',
    
    # GEPA convenience optimizers
    'GEPAMute', 'GEPAAdaptive',
]
