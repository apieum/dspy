"""Generic strategy contract for Darwin optimizers."""

from abc import ABC
from typing import Any, Generic, TypeVar

from ..compilation_observer import CompilationObserver, LoggingCompilationObserver
from ..result import Result
from ..workflow import Workflow

R = TypeVar("R", bound=Result)


class BaseStrategy(Workflow[R], ABC, Generic[R]):
    """Public strategy adapter contract.

    A strategy selects and configures a workflow.  Algorithm-specific state,
    components, stopping rules, and phase transitions belong to that
    workflow, not to this shared base class.
    """

    def __init__(self, config) -> None:
        super().__init__(config, notify=self._notify)
        self.observers: tuple[CompilationObserver, ...] = (
            *tuple(config.observers),
            LoggingCompilationObserver(verbose=config.verbose),
        )

    def _notify(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Publish lifecycle events at the strategy boundary."""
        for observer in self.observers:
            callback = getattr(observer, event, None)
            if callback is not None:
                callback(*args, **kwargs)
