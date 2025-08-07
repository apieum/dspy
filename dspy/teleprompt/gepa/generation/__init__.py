"""Generation step of GEPA optimization - DSPy Native Implementation."""

from ..data.cohort import Cohort
from .generator import Generator
from .crossover import SystemAwareMerge

# DSPy-native implementation (reuses DSPy's built-in systems)
from .feedback import FeedbackProvider
from .reflective_mutation_native import ReflectivePromptMutation
from .reflection_strategy import (
    ReflectionStrategy, 
    GEPAReflection, SimpleReflection, PrefixReflection
)
from .prompt_mutator import (
    PromptMutator,
    ReflectivePromptMutator, SimplePromptMutator, NoOpMutator
)
from .evolvable_module import EvolvableModule

__all__ = [
    # Core components
    'Cohort', 
    'Generator', 
    'ReflectivePromptMutation', 
    'FeedbackProvider',
    'SystemAwareMerge',
    
    # DSPy-native architectural components  
    'ReflectionStrategy',
    'GEPAReflection', 'SimpleReflection', 'PrefixReflection',
    'PromptMutator',
    'ReflectivePromptMutator', 'SimplePromptMutator', 'NoOpMutator',
    'EvolvableModule'
]