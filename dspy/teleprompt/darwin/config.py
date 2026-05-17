"""Configuration classes for Darwin optimization framework."""

from dataclasses import dataclass, field
from typing import List, Type, Callable, Optional
from abc import ABC

from .budget import Budget, LMCallsBudget
from .selection import Selector, ParetoFrontier
from .generation import Generator, SystemAwareMerge, ReflectivePromptMutation
from .evaluation import Evaluator, GEPATwoPhasesEval
from .evaluation.metrics import Assessor, F1Score
from .strategy import BaseStrategy
from .observers import ChannelContext


@dataclass
class DarwinConfig:
    """Minimal configuration with only component classes and strategic choices.

    Strategy owns all tactical decisions and instantiates components with appropriate data.
    Config only specifies which classes to use and key strategic choices like metrics.
    """
    # Strategic components for strategy to instantiate (required)
    budget: Type['Budget'] = LMCallsBudget
    selection: Type['Selector'] = ParetoFrontier
    evaluation: Type['Evaluator'] = GEPATwoPhasesEval
    mutation: Type['Generator'] = ReflectivePromptMutation
    fitness_function: Assessor = F1Score()

    # Optional strategic choices
    crossover: Optional[Type['Generator']] = SystemAwareMerge  # Optional crossover generator
    enhanced_feedback: Optional[Assessor] = F1Score()  # Optional feedback-generating metric

    # System parameters we actually have
    max_lm_calls: int = 100
    patience: int = 3
    validation_split: float = 0.2
    minibatch_size: int = 3  # Size of minibatch for quick validation
    seed: int = 1

    # Logging
    verbose: bool = False