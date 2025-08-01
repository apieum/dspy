"""Data structures for GEPA optimization."""

from .structures import ScoreMatrix, FeedbackResult, EvaluationTrace, ModuleFeedback
from .candidate_pool import CandidatePool, CandidateLineage
from .dataset import TrainingDataset, SplitDataset

__all__ = [
    'ScoreMatrix',
    'FeedbackResult', 
    'EvaluationTrace',
    'ModuleFeedback',
    'CandidatePool',
    'CandidateLineage',
    'TrainingDataset',
    'SplitDataset'
]