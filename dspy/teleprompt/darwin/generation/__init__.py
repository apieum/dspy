"""Generation step of GEPA optimization - DSPy Native Implementation."""

from ..data.cohort import Cohort
from .generator import Generator
from .system_aware_merge import SystemAwareMerge

# DSPy-native implementation (reuses DSPy's built-in systems)
from .feedback import FeedbackProvider
from .config import ReflectiveMutationConfig, ModuleSelectionStrategy
from .enhanced_metrics import (
    CodeEvaluationAssessor
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

__all__ = [
    # Core components
    'Cohort',
    'Generator',
    'ReflectivePromptMutation',
    'FeedbackProvider',
    'SystemAwareMerge',
    'ReflectiveMutationConfig',
    'ModuleSelectionStrategy',

    # Enhanced μf-compliant assessors
    'CodeEvaluationAssessor',

    # DSPy-native architectural components
    'ReflectionStrategy',
    'GEPAReflection',
    'PromptMutator',
    'ReflectivePromptMutator',
    'EvolvableModule'
]
