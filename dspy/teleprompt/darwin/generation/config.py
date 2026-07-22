"""Configuration objects for Darwin generation components."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class ModuleSelectionStrategy(Enum):
    """Strategy for selecting which predictor to mutate."""

    RANDOM = "random"
    WORST_PERFORMING = "worst_performing"
    ALL = "all"
    ROUND_ROBIN = "round_robin"


@dataclass
class ReflectiveMutationConfig:
    """Tunable options for reflective prompt mutation."""

    minibatch_size: int = 5
    module_selection_strategy: ModuleSelectionStrategy = ModuleSelectionStrategy.WORST_PERFORMING
    max_retries: int = 3
    reflection_strategy: Optional["ReflectionStrategy"] = None
    feedback_provider: Optional["FeedbackProvider"] = None
    enhanced_feedback_function: Optional[Callable[..., Any]] = None
    selection_temperature: float = 1.0
    max_modules_per_generation: Optional[int] = None
    enable_detailed_logging: bool = False
    preserve_original_on_failure: bool = True
    use_abstract_feedback: bool = False

    def __post_init__(self):
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.selection_temperature <= 0.0:
            raise ValueError("selection_temperature must be positive")
        if self.max_modules_per_generation is not None and self.max_modules_per_generation <= 0:
            raise ValueError("max_modules_per_generation must be positive or None")

    @classmethod
    def for_quick_experiments(cls, **overrides: Any) -> "ReflectiveMutationConfig":
        defaults = {
            "minibatch_size": 3,
            "max_retries": 1,
            "module_selection_strategy": ModuleSelectionStrategy.RANDOM,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_production(cls, **overrides: Any) -> "ReflectiveMutationConfig":
        defaults = {
            "minibatch_size": 8,
            "max_retries": 5,
            "module_selection_strategy": ModuleSelectionStrategy.WORST_PERFORMING,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_debugging(cls, **overrides: Any) -> "ReflectiveMutationConfig":
        defaults = {
            "enable_detailed_logging": True,
            "minibatch_size": 2,
            "max_retries": 1,
            "preserve_original_on_failure": True,
        }
        defaults.update(overrides)
        return cls(**defaults)
