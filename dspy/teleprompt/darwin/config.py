"""Configuration contract for the Darwin execution framework."""

from abc import ABC
from typing import Any, Optional, Tuple, Type

from .budget import Budget


class DarwinConfig(ABC):
    """Minimal lifecycle configuration required by Darwin strategies."""

    budget: Type[Budget]
    observers: Tuple[Any, ...]
    verbose: bool
    checkpoint_path: Optional[str]
    resume_from: Optional[str]
    handle_signals: bool
