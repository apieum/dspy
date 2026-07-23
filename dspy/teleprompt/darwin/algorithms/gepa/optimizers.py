"""GEPA optimizer implementations using the Darwin framework.

These are convenience classes that configure Darwin with specific GEPA strategies
as described in the GEPA paper.
"""

import inspect
from typing import Optional, Callable, Any
from ...optimizer import Darwin
from .candidate import example_id
from .generation.mutation import ReflectivePromptMutation
from .adaptive_generator import GEPAAdaptiveGenerator
from .mutation_config import ReflectiveMutationConfig
from ...evaluation.metrics import Metric
from .config import GEPAConfig
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

        def assessor(example, prediction, trace=None, pred_name=None, pred_trace=None):
            args = (example, prediction, trace, pred_name, pred_trace)
            # Preserve the official GEPA metric contract when the caller
            # supplies predictor-level context, while retaining old metrics.
            positional_count = min(
                5,
                len([
                    p for p in signature.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]),
            )
            result = metric(*args[:positional_count])

            if isinstance(result, Metric):
                return Metric(
                    result.value,
                    id=example_id(example),
                    feedback=result.feedback,
                    errors=result.errors,
                    suggestions=result.suggestions,
                    trace=result.trace if result.trace is not None else trace,
                    objective_scores=result.objective_scores,
                    side_info=result.side_info,
                )
            if isinstance(result, tuple) and len(result) == 2:
                value, feedback = result
                return Metric(value, id=example_id(example), feedback=str(feedback), trace=trace)
            return Metric(result, id=example_id(example), trace=trace)

        return assessor

    raise TypeError("metric must be callable")


class GEPAMute(Darwin):
    """GEPA implementation with reflective mutation generation.
    
    Reflective mutation is the primary generation strategy. Opportunistic
    merging follows ``use_merge`` and is enabled by default for GEPA parity.
    """
    
    def __init__(
        self,
        metric: Callable[[Any, Any, Optional[Any]], float],
        max_calls: int = 1000,
        minibatch_size: int = 25,
        patience: Optional[int] = None,
        verbose: bool = False,
        reflection_strategy=None,
        feedback_function: Optional[Callable[..., Any]] = None,
        seed: int = 1,
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
        config = GEPAConfig(
            max_lm_calls=max_calls,
            minibatch_size=minibatch_size,
            patience=patience,
            verbose=verbose,
            seed=seed,
            mutation=ReflectivePromptMutation,
            fitness_function=assessor,
            enhanced_feedback=assessor,
            mutation_config=ReflectiveMutationConfig(
                minibatch_size=minibatch_size,
                reflection_strategy=reflection_strategy,
                enhanced_feedback_function=feedback_function,
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
        config = GEPAConfig(
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
