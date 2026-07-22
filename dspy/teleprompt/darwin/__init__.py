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
from .generation import (
    Generator,
    SamplingStrategy,
    BatchSampler,
    EpochShuffledBatchSampler,
    SingleMutationSampling,
    SameParentSampling,
    IndependentSampling,
    PxNSampling,
)
from .evaluation import Evaluator

# Business step implementations
from .budget import LMCallsBudget, IterationBudget, AdaptiveBudget
from .selection.pareto import ParetoFrontier
from .selection.archive import DiversityArchive
from .generation.mutation import ReflectivePromptMutation
from .generation.system_aware_merge import SystemAwareMerge
from .generation.feedback import FeedbackProvider
from .evaluation.gepa_evaluator import FullTaskScores, ParentFastCompare, GEPATwoPhasesEval
from .evaluation.acceptance import StrictImprovementAcceptance, ImprovementOrEqualAcceptance
from .evaluation.cache import EvaluationCache
from .evaluation.proposal_selection import AllImprovements, BestImprovement, TopKImprovements
from .evaluation.policy import EvaluationPolicy, FullEvaluationPolicy, MinibatchEvaluationPolicy
from .evaluation.trace_collector import EnhancedTraceCollector
from .evaluation.feedback import FeedbackResult, EvaluationTrace, ModuleFeedback
from .evaluation.metrics import (
    Metric,
    BaseAssessor,
    ExactMatch,
    Contains,
    F1Score,
    RougeL,
    Bleu,
    CustomMetric,
    CompositeMetric,
)

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
from .compilation_observer import CompilationObserver
from .dataset_manager import (
    DatasetManager,
    DatasetManagerFactory,
    DefaultDatasetManager,
    DefaultDatasetManagerFactory,
)
from .generation.config import ReflectiveMutationConfig, ModuleSelectionStrategy
from .state import OptimizationCheckpoint

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
    'SamplingStrategy',
    'BatchSampler',
    'EpochShuffledBatchSampler',
    'SingleMutationSampling',
    'SameParentSampling',
    'IndependentSampling',
    'PxNSampling',
    'Evaluator',

    # Business step implementations
    'LMCallsBudget',
    'IterationBudget',
    'AdaptiveBudget',
    'ParetoFrontier',
    'DiversityArchive',
    'ReflectivePromptMutation',
    'SystemAwareMerge',
    'FeedbackProvider',
    'FullTaskScores',
    'ParentFastCompare',
    'StrictImprovementAcceptance',
    'ImprovementOrEqualAcceptance',
    'EvaluationCache',
    'AllImprovements',
    'BestImprovement',
    'TopKImprovements',
    'EvaluationPolicy',
    'FullEvaluationPolicy',
    'MinibatchEvaluationPolicy',
    'OptimizationCheckpoint',
    'GEPATwoPhasesEval',
    'EnhancedTraceCollector',
    'FeedbackResult',
    'EvaluationTrace',
    'ModuleFeedback',
    'Metric',
    'BaseAssessor',
    'ExactMatch',
    'Contains',
    'F1Score',
    'RougeL',
    'Bleu',
    'CustomMetric',
    'CompositeMetric',

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
    'CompilationObserver',
    'DatasetManager', 'DatasetManagerFactory',
    'DefaultDatasetManager', 'DefaultDatasetManagerFactory',
    'ReflectiveMutationConfig',
    'ModuleSelectionStrategy',
    
    # GEPA convenience optimizers
    'GEPAMute', 'GEPAAdaptive',
]
