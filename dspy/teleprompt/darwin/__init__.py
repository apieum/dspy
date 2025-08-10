"""Darwin - Extensible evolutionary optimization for language model programs.

Darwin is a general-purpose toolkit for building evolutionary optimizers,
with GEPA as the first "recipe" using the framework.
"""

# Main Darwin implementation
from .optimizer import Darwin, GEPA, GEPAMerge, GEPAMute
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
from .evaluation.gepa_evaluator import FullTaskScores, ParentFastCompare
from .evaluation.trace_collector import EnhancedTraceCollector
from .evaluation.feedback import FeedbackResult, EvaluationTrace, ModuleFeedback

# Factory functions are now static methods on Darwin class

__all__ = [
    # Core classes
    'Darwin',
    'GEPA', 'GEPAMerge', 'GEPAMute',
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
    'FullTaskScores',
    'ParentFastCompare',
    'EnhancedTraceCollector',
    'FeedbackResult',
    'EvaluationTrace',
    'ModuleFeedback',
]
