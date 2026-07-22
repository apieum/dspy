"""GEPA optimizer implementations using the Darwin framework.

These are convenience classes that configure Darwin with specific GEPA strategies
as described in the GEPA paper.
"""

import inspect
from typing import Optional, Callable, Any
from .optimizer import Darwin
from .data.candidate import example_id
from .generation.mutation import ReflectivePromptMutation
from .generation.adaptive_generator import GEPAAdaptiveGenerator
from .generation.config import ReflectiveMutationConfig
from .evaluation.metrics import Metric
from .config import DarwinConfig
from .strategy import GEPAStrategy


def _as_assessor(metric: Callable[[Any, Any, Optional[Any]], float]):
    """Adapt old-style metric callables to Darwin's assessor signature."""
    if isinstance(metric, type) or hasattr(metric, "__call__"):
        try:
            signature = inspect.signature(metric)
            parameter_count = len(
                [
                    p
                    for p in signature.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty
                ]
            )
        except (TypeError, ValueError):
            parameter_count = 3

        def assessor(example, prediction, trace=None):
            if parameter_count <= 2:
                result = metric(example, prediction)
            else:
                result = metric(example, prediction, trace)

            if isinstance(result, Metric):
                return result
            if isinstance(result, tuple) and len(result) == 2:
                value, feedback = result
                return Metric(value, id=example_id(example), feedback=str(feedback), trace=trace)
            return Metric(result, id=example_id(example), trace=trace)

        return assessor

    raise TypeError("metric must be callable")


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
        reflection_strategy=None,
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
        assessor = _as_assessor(metric)
        config = DarwinConfig(
            max_lm_calls=max_calls,
            minibatch_size=minibatch_size,
            patience=patience,
            verbose=verbose,
            mutation=ReflectivePromptMutation,
            fitness_function=assessor,
            enhanced_feedback=assessor,
            mutation_config=ReflectiveMutationConfig(
                minibatch_size=minibatch_size,
                reflection_strategy=reflection_strategy,
            ),
        )

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
        assessor = _as_assessor(metric)
        config = DarwinConfig(
            max_lm_calls=max_calls,
            minibatch_size=minibatch_size,
            patience=patience,
            verbose=verbose,
            mutation=GEPAAdaptiveGenerator,
            fitness_function=assessor,
            enhanced_feedback=assessor,
            mutation_config=ReflectiveMutationConfig(minibatch_size=minibatch_size),
        )

        super().__init__(GEPAStrategy, config, **kwargs)
