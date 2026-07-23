"""Generic evaluation contracts and utilities."""

from .evaluator import Evaluator
from .batching import BatchEvaluator, PerCandidateBatchEvaluator, CallbackBatchEvaluator
from .acceptance import StrictImprovementAcceptance, ImprovementOrEqualAcceptance
from .cache import EvaluationCache
from .proposal_selection import AllImprovements, BestImprovement, TopKImprovements
from .policy import EvaluationPolicy, FullEvaluationPolicy, MinibatchEvaluationPolicy
from .baseline_comparison import BaselineComparison
from .comprehensive import ComprehensiveEvaluator
from .feedback import FeedbackProvider
from .metrics import (
    Assessor,
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
from .confidence import ConfidenceAssessor, LinearConfidenceScoring, ThresholdConfidenceScoring, extract_logprob

__all__ = [
    'Evaluator',
    'BatchEvaluator',
    'PerCandidateBatchEvaluator',
    'CallbackBatchEvaluator',
    'BaselineComparison',
    'ComprehensiveEvaluator',
    'FeedbackProvider',
    'StrictImprovementAcceptance',
    'ImprovementOrEqualAcceptance',
    'EvaluationCache',
    'AllImprovements',
    'BestImprovement',
    'TopKImprovements',
    'EvaluationPolicy',
    'FullEvaluationPolicy',
    'MinibatchEvaluationPolicy',
    'Assessor',
    'Metric',
    'BaseAssessor',
    'ExactMatch',
    'Contains',
    'F1Score',
    'RougeL',
    'Bleu',
    'CustomMetric',
    'CompositeMetric',
    'ConfidenceAssessor',
    'LinearConfidenceScoring',
    'ThresholdConfidenceScoring',
    'extract_logprob',
]
