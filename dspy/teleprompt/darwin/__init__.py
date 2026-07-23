"""Darwin framework for composing and executing optimization strategies.

Algorithm recipes such as GEPA live under ``darwin.algorithms`` and are not
imported by this module.  This keeps the framework independent from any one
optimization algorithm.
"""

from .optimizer import Darwin
from .config import DarwinConfig
from .data.candidate import Candidate
from .data.cohort import Cohort, Survivors, Parents, NewBorns
from .budget import (
    Budget,
    BudgetStrategy,
    BudgetEvent,
    BudgetExhaustedError,
    LMCallsBudget,
    IterationBudget,
    AdaptiveBudget,
)
from .selection import Selector
from .generation import (
    Generator,
    SamplingStrategy,
    ProposalTask,
    BatchSampler,
    EpochShuffledBatchSampler,
    SingleMutationSampling,
    SameParentSampling,
    IndependentSampling,
    PxNSampling,
)
from .evaluation import (
    Evaluator,
    BatchEvaluator,
    PerCandidateBatchEvaluator,
    CallbackBatchEvaluator,
    StrictImprovementAcceptance,
    ImprovementOrEqualAcceptance,
    EvaluationCache,
    AllImprovements,
    BestImprovement,
    TopKImprovements,
    EvaluationPolicy,
    FullEvaluationPolicy,
    MinibatchEvaluationPolicy,
    Metric,
    BaseAssessor,
    ExactMatch,
    Contains,
    F1Score,
    RougeL,
    Bleu,
    CustomMetric,
    CompositeMetric,
    ConfidenceAssessor,
    LinearConfidenceScoring,
    ThresholdConfidenceScoring,
    extract_logprob,
)
from .result import Result, OptimizationFailureError
from .observers import (
    OptimizerObserver,
    SelectorObserver,
    GeneratorObserver,
    EvaluatorObserver,
    CandidateObserver,
    ChannelContext,
    Channel,
)
from .logging import (
    AsyncLogger,
    EvaluationLogger,
    SelectionLogger,
    GenerationLogger,
    StrategyLogger,
    LoggerFactory,
)
from .strategy import BaseStrategy
from .compilation_observer import CompilationObserver, LoggingCompilationObserver
from .data.dataset_manager import (
    DatasetManager,
    DatasetManagerFactory,
    DefaultDatasetManager,
    DefaultDatasetManagerFactory,
)
from .state import Checkpointable, OptimizationCheckpoint
from .stopping import Stopper, ScoreThresholdStopper, NoImprovementStopper, FileStopper, AnyStopper

__all__ = [
    "Darwin", "DarwinConfig", "Candidate", "Cohort", "Survivors", "Parents", "NewBorns",
    "Budget", "BudgetStrategy", "BudgetEvent", "BudgetExhaustedError",
    "LMCallsBudget", "IterationBudget", "AdaptiveBudget", "Selector", "Generator",
    "SamplingStrategy", "ProposalTask", "BatchSampler", "EpochShuffledBatchSampler",
    "SingleMutationSampling", "SameParentSampling", "IndependentSampling", "PxNSampling",
    "Evaluator", "BatchEvaluator", "PerCandidateBatchEvaluator", "CallbackBatchEvaluator",
    "StrictImprovementAcceptance", "ImprovementOrEqualAcceptance", "EvaluationCache",
    "AllImprovements", "BestImprovement", "TopKImprovements", "EvaluationPolicy",
    "FullEvaluationPolicy", "MinibatchEvaluationPolicy", "Metric", "BaseAssessor",
    "ExactMatch", "Contains", "F1Score", "RougeL", "Bleu", "CustomMetric", "CompositeMetric",
    "ConfidenceAssessor", "LinearConfidenceScoring", "ThresholdConfidenceScoring",
    "extract_logprob", "Result", "OptimizationFailureError", "Checkpointable",
    "OptimizationCheckpoint", "BaseStrategy", "CompilationObserver", "LoggingCompilationObserver",
    "DatasetManager", "DatasetManagerFactory", "DefaultDatasetManager", "DefaultDatasetManagerFactory",
    "OptimizerObserver", "SelectorObserver", "GeneratorObserver", "EvaluatorObserver",
    "CandidateObserver", "ChannelContext", "Channel", "AsyncLogger", "EvaluationLogger",
    "SelectionLogger", "GenerationLogger", "StrategyLogger", "LoggerFactory", "Stopper",
    "ScoreThresholdStopper", "NoImprovementStopper", "FileStopper", "AnyStopper",
]
