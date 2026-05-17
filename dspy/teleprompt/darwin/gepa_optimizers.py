"""GEPA optimizer implementations using the Darwin framework.

These are convenience classes that configure Darwin with specific GEPA strategies
as described in the GEPA paper.
"""

from typing import Optional, Callable, Any, List
from dspy import Example
from .optimizer import Darwin
from .budget import LMCallsBudget
from .selection.pareto import ParetoFrontier
from .generation.mutation import ReflectivePromptMutation
from .generation.adaptive_generator import GEPAAdaptiveGenerator
from .generation.feedback import FeedbackProvider
from .evaluation.gepa_evaluator import FullTaskScores
from .evaluation.metrics import F1Score
from .config import DarwinConfig
from .strategy import GEPAStrategy


class GEPAMute(Darwin):
    """GEPA implementation with mutation-only generation strategy.
    
    This implements the standard GEPA algorithm using only reflective prompt
    mutation for candidate generation, without opportunistic merging.
    """
    
    def __init__(
        self,
        metric: Callable[[Any, Any, Optional[Any]], float],
        max_calls: int = 1000,
        minibatch_size: int = 25,
        patience: int = 5,
        verbose: bool = False,
        **kwargs
    ):
        """Initialize GEPAMute optimizer.
        
        Args:
            metric: Evaluation metric function
            max_calls: Maximum number of LLM calls
            minibatch_size: Size of minibatches for evaluation
            patience: Generations without progress before termination
            verbose: Enable detailed logging
            **kwargs: Additional arguments passed to Darwin
        """
        # Create config with GEPA settings
        config = DarwinConfig(
            max_lm_calls=max_calls,
            minibatch_size=minibatch_size,
            patience=patience,
            verbose=verbose
        )
        
        # Create feedback provider
        feedback_provider = FeedbackProvider(F1Score())
        
        # Override generation component with feedback provider
        config.mutation = lambda: ReflectivePromptMutation(feedback_provider)
        
        super().__init__(GEPAStrategy, config, **kwargs)


class GEPAAdaptive(Darwin):
    """GEPA implementation with adaptive generation strategy.
    
    This implements the adaptive GEPA algorithm that combines reflective prompt
    mutation with opportunistic merging based on performance feedback.
    """
    
    def __init__(
        self,
        metric: Callable[[Any, Any, Optional[Any]], float],
        max_calls: int = 1000,
        minibatch_size: int = 25,
        patience: int = 5,
        verbose: bool = False,
        **kwargs
    ):
        """Initialize GEPAAdaptive optimizer.
        
        Args:
            metric: Evaluation metric function
            max_calls: Maximum number of LLM calls
            minibatch_size: Size of minibatches for evaluation
            patience: Generations without progress before termination
            verbose: Enable detailed logging
            **kwargs: Additional arguments passed to Darwin
        """
        # Create config with GEPA settings
        config = DarwinConfig(
            max_lm_calls=max_calls,
            minibatch_size=minibatch_size,
            patience=patience,
            verbose=verbose
        )
        
        # Override generation component with adaptive generator
        config.mutation = GEPAAdaptiveGenerator
        
        super().__init__(GEPAStrategy, config, **kwargs)