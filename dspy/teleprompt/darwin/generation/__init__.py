"""Generation step of GEPA optimization - DSPy Native Implementation."""

from ..data.cohort import Cohort
from .generator import Generator
from .system_aware_merge import SystemAwareMerge

# DSPy-native implementation (reuses DSPy's built-in systems)
from .feedback import FeedbackProvider
from .enhanced_metrics import (
    code_evaluation_metric, math_problem_metric,
    text_classification_metric, qa_accuracy_metric
)
from .mutation import ReflectivePromptMutation
from .reflection_strategy import (
    ReflectionStrategy,
    GEPAReflection
)
from .prompt_mutator import (
    PromptMutator,
    ReflectivePromptMutator
)
from .evolvable_module import EvolvableModule
from .config import ReflectiveMutationConfig, ModuleSelectionStrategy

__all__ = [
    # Core components
    'Cohort',
    'Generator',
    'ReflectivePromptMutation',
    'FeedbackProvider',
    'SystemAwareMerge',

    # Configuration
    'ReflectiveMutationConfig',
    'ModuleSelectionStrategy',

    # Enhanced μf-compliant metrics
    'code_evaluation_metric', 'math_problem_metric',
    'text_classification_metric', 'qa_accuracy_metric',

    # DSPy-native architectural components
    'ReflectionStrategy',
    'GEPAReflection',
    'PromptMutator',
    'ReflectivePromptMutator',
    'EvolvableModule'
]
