"""Base lifecycle contract for Darwin optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import dspy

from ..compilation_observer import CompilationObserver, LoggingCompilationObserver
from ..result import OptimizationFailureError, Result
from dspy.teleprompt.teleprompt import Teleprompter

R = TypeVar("R", bound=Result)


class BaseStrategy(Teleprompter, ABC, Generic[R]):
    """Common strategy executor with a stable, observable lifecycle.

    Public lifecycle methods deliberately live here.  Subclasses implement
    only the protected hooks, so every strategy receives the same observer
    notifications and execution semantics.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.latest_result: R | None = None
        self.observers: tuple[CompilationObserver, ...] = (
            *tuple(config.observers),
            LoggingCompilationObserver(verbose=config.verbose),
        )

    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        teacher: dspy.Module | None = None,
        valset: list[dspy.Example] | None = None,
        devset: list[dspy.Example] | None = None,
        **kwargs: Any,
    ) -> dspy.Module:
        """Run the strategy through its observable lifecycle."""
        if valset is not None and devset is not None:
            raise ValueError("Pass either valset or devset, not both")
        validation_set = valset if valset is not None else devset
        try:
            self.start_compilation(
                student=student,
                trainset=trainset,
                devset=validation_set,
                teacher=teacher,
                **kwargs,
            )
            while self.next_step():
                pass
            result = self.finish_compilation()
            self.latest_result = result
            return self._compile_result(result)
        except OptimizationFailureError:
            raise
        except Exception as error:
            raise OptimizationFailureError(str(error)) from error

    def get_last_result(self) -> R | None:
        return self.latest_result

    def _notify(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Publish an event to configured observers."""
        for observer in self.observers:
            callback = getattr(observer, event, None)
            if callback is not None:
                callback(*args, **kwargs)

    def start_compilation(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        devset: list[dspy.Example] | None = None,
        teacher: dspy.Module | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a compilation and notify observers after setup completes."""
        self._start_compilation(
            student,
            trainset=trainset,
            devset=devset,
            teacher=teacher,
            **kwargs,
        )
        self._notify(
            "start_compilation",
            student,
            self._dataset_manager_for_notification(),
        )

    def next_step(self) -> bool:
        """Execute one strategy step and always notify observers."""
        continuing = self._next_step()
        self._notify("next_step", self, continuing)
        return continuing

    def finish_compilation(self) -> R:
        """Finalize a compilation and notify observers with the result module."""
        result = self._finish_compilation()
        self._notify("finish_compilation", self._result_module_for_notification(result))
        return result

    def _dataset_manager_for_notification(self):
        """Return the active dataset manager without knowing strategy state."""
        return getattr(self, "dataset_manager", None)

    @abstractmethod
    def _result_module_for_notification(self, result: R) -> dspy.Module | None:
        """Provide the module sent with the generic finish notification."""
        raise NotImplementedError

    @abstractmethod
    def _start_compilation(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        devset: list[dspy.Example] | None = None,
        teacher: dspy.Module | None = None,
        **kwargs: Any,
    ) -> None:
        """Implement compilation setup for the concrete strategy."""
        raise NotImplementedError

    @abstractmethod
    def _next_step(self) -> bool:
        """Implement one concrete strategy step."""
        raise NotImplementedError

    @abstractmethod
    def _finish_compilation(self) -> R:
        """Implement concrete strategy finalization."""
        raise NotImplementedError

    @abstractmethod
    def _compile_result(self, result: R) -> dspy.Module:
        """Convert an algorithm result into the compiled module it returns."""
        raise NotImplementedError
