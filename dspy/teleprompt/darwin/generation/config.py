"""Configuration objects for GEPA generation components.

This module provides configuration classes that make GEPA generators more
modular, testable, and easier to experiment with.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any, List
from enum import Enum

import dspy


class ModuleSelectionStrategy(Enum):
    """Strategy for selecting which modules to mutate."""
    RANDOM = "random"
    WORST_PERFORMING = "worst_performing" 
    ALL = "all"
    ROUND_ROBIN = "round_robin"


@dataclass
class ReflectiveMutationConfig:
    """Configuration for ReflectivePromptMutation behavior.
    
    This configuration object makes the generator's behavior transparent
    and easily tunable for experimentation without code changes.
    
    Usage:
        config = ReflectiveMutationConfig(
            minibatch_size=3,
            module_selection_strategy=ModuleSelectionStrategy.WORST_PERFORMING,
            max_retries=2
        )
        generator = ReflectivePromptMutation(config=config)
    """
    
    # Core generation parameters
    minibatch_size: int = 5
    """Number of feedback examples to use for each mutation."""
    
    module_selection_strategy: ModuleSelectionStrategy = ModuleSelectionStrategy.WORST_PERFORMING
    """Strategy for selecting which predictors to mutate."""
    
    
    max_retries: int = 3
    """Maximum attempts to generate improved instructions."""
    
    # Reflection configuration  
    reflection_strategy: Optional[Any] = None
    """ReflectionStrategy instance for generating improved instructions."""
    
    # Feedback configuration
    feedback_provider: Optional[Any] = None
    """FeedbackProvider instance for evaluation and diagnostics."""
    
    enhanced_feedback_function: Optional[Callable] = None
    """Enhanced μf function for rich diagnostic feedback."""
    
    # Module selection parameters
    selection_temperature: float = 1.0
    """Temperature for probabilistic module selection strategies."""
    
    max_modules_per_generation: Optional[int] = None
    """Maximum number of modules to mutate per generation (None = no limit)."""
    
    # Debugging and observability
    enable_detailed_logging: bool = False
    """Enable detailed logging for debugging mutation process."""
    
    preserve_original_on_failure: bool = True
    """Keep original module if all mutation attempts fail."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.selection_temperature <= 0.0:
            raise ValueError("selection_temperature must be positive")
        
        if (self.max_modules_per_generation is not None and 
            self.max_modules_per_generation <= 0):
            raise ValueError("max_modules_per_generation must be positive or None")
    
    @classmethod
    def for_quick_experiments(cls, **overrides) -> 'ReflectiveMutationConfig':
        """Create configuration optimized for quick experimental iterations."""
        defaults = {
            'minibatch_size': 3,
            'max_retries': 1,
            'module_selection_strategy': ModuleSelectionStrategy.RANDOM,
        }
        defaults.update(overrides)
        return cls(**defaults)
    
    @classmethod
    def for_production(cls, **overrides) -> 'ReflectiveMutationConfig':
        """Create configuration optimized for production performance."""
        defaults = {
            'minibatch_size': 8,
            'max_retries': 5,
            'module_selection_strategy': ModuleSelectionStrategy.WORST_PERFORMING,
        }
        defaults.update(overrides)
        return cls(**defaults)
    
    @classmethod  
    def for_debugging(cls, **overrides) -> 'ReflectiveMutationConfig':
        """Create configuration with detailed logging for debugging."""
        defaults = {
            'enable_detailed_logging': True,
            'minibatch_size': 2,  # Small for faster debugging
            'max_retries': 1,
            'preserve_original_on_failure': True,
        }
        defaults.update(overrides)
        return cls(**defaults)