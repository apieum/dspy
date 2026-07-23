"""Generic workflow runtime used by Darwin strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TYPE_CHECKING, TypeVar

import dspy

if TYPE_CHECKING:
    from .compilation_observer import CompilationObserver
    from .config import DarwinConfig
    from .result import Result

R = TypeVar("R", bound="Result")


class Workflow(ABC, Generic[R]):
    """Small execution contract independent of any optimization algorithm.

    A workflow owns the algorithm's runtime state and decides which operation
    runs next.  The strategy layer can expose it through a public adapter
    without knowing the workflow's domain objects.
    """

    def __init__(
        self,
        config: "DarwinConfig",
        *,
        notify: Callable[..., None] | None = None,
    ) -> None:
        self.config = config
        self._notify_callback = notify

    def notify(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an optional lifecycle event to the strategy boundary."""
        if self._notify_callback is not None:
            self._notify_callback(event, *args, **kwargs)

    @abstractmethod
    def start_compilation(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        devset: list[dspy.Example] | None = None,
        teacher: dspy.Module | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_step(self) -> bool:
        """Execute one workflow step and report whether execution continues."""
        raise NotImplementedError

    @abstractmethod
    def finish_compilation(self) -> R:
        raise NotImplementedError
