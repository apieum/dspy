"""Base lifecycle and execution contract for Darwin strategies."""

from __future__ import annotations

import json
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import dspy
from dspy.teleprompt.teleprompt import Teleprompter

from ..compilation_observer import CompilationObserver, LoggingCompilationObserver
from ..result import OptimizationFailureError, Result
from ..budget import BudgetEvent, BudgetExhaustedError
from ..state import OptimizationCheckpoint

R = TypeVar("R", bound=Result)


class BaseStrategy(Teleprompter, ABC, Generic[R]):
    """Generic strategy executor with a stable, observable lifecycle."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.latest_result: R | None = None
        self.observers: tuple[CompilationObserver, ...] = (
            *tuple(config.observers),
            LoggingCompilationObserver(verbose=config.verbose),
        )
        self.budget = config.budget(config=config)
        self._stop_requested = False
        self._stop_reason = None
        self._previous_signal_handlers = {}
        self._signal_stop_requested = False
        self._signal_stop_reason = None
        self.initialize()

    def initialize(self) -> None:
        """Initialize strategy-owned state after the base contract is ready."""

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

    def request_stop(self, reason: str) -> None:
        """Accept a stop request from a budget or another lifecycle peer."""
        if self._stop_requested:
            return
        self._stop_requested = True
        self._stop_reason = reason
        if reason == "budget_exhausted":
            self._notify("budget_exhausted", self.budget)

    def start_compilation(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        devset: list[dspy.Example] | None = None,
        teacher: dspy.Module | None = None,
        **kwargs: Any,
    ) -> None:
        self.budget = self.config.budget(config=self.config)
        self._stop_requested = False
        self._stop_reason = None
        self._signal_stop_requested = False
        self._signal_stop_reason = None
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
        self.budget.reconcile(self)
        continuing = not (
            self._stop_requested or self._signal_stop_requested
        )
        if continuing:
            continuing = self._next_step()
        self._notify("next_step", self, continuing)
        return continuing

    def start_iteration(self, iteration: int, cohort) -> None:
        self._notify("start_iteration", iteration, cohort, self.budget)

    def finish_iteration(self, iteration: int, cohort) -> None:
        try:
            self.budget.spend(
                BudgetEvent("iteration", metadata={"iteration": iteration})
            )
        except BudgetExhaustedError:
            self.request_stop("budget_exhausted")
        self._notify("finish_iteration", iteration, cohort, self.budget)

    def finish_compilation(self) -> R:
        result = self._finish_compilation()
        self._notify("finish_compilation", result)
        return result

    def _dataset_manager_for_notification(self):
        return getattr(self, "dataset_manager", None)

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
        raise NotImplementedError

    @abstractmethod
    def _next_step(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _finish_compilation(self) -> R:
        raise NotImplementedError

    @abstractmethod
    def _compile_result(self, result: R) -> dspy.Module:
        raise NotImplementedError

    # Generic checkpoint envelope and signal handling.  Algorithms provide
    # only their serializable domain state through the hooks below.
    def get_checkpoint(self, completed: bool = False) -> OptimizationCheckpoint:
        return OptimizationCheckpoint(
            generation=getattr(self, "current_generation", 0),
            algorithm_state=getattr(self, "algorithm_state", "initialize"),
            history=list(getattr(self, "history", [])),
            budget=self.budget.serialize_state(),
            candidates=self._checkpoint_candidates(),
            strategy_state=self._get_checkpoint_strategy_state(),
            rng_state=self._checkpoint_rng_state(),
            stop_reason=self._stop_reason or self._signal_stop_reason,
            completed=completed,
        )

    def _checkpoint_candidates(self) -> list[dict]:
        return []

    def _get_checkpoint_strategy_state(self) -> dict:
        return {}

    def _checkpoint_rng_state(self):
        return None

    def _restore_checkpoint(self, path: str, student: dspy.Module) -> bool:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Darwin checkpoint not found: {path}")
        checkpoint = OptimizationCheckpoint.from_dict(
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
        )
        self.budget = type(self.budget).restore_state(checkpoint.budget)
        return self._restore_checkpoint_state(checkpoint, student)

    def _restore_checkpoint_state(
        self, checkpoint: OptimizationCheckpoint, student: dspy.Module
    ) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__} does not support checkpoint restoration"
        )

    def _write_checkpoint(self, completed: bool = False) -> None:
        checkpoint_path = self.config.checkpoint_path
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.get_checkpoint(completed=completed).to_dict(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _install_signal_handlers(self) -> None:
        if not self.config.handle_signals:
            return
        self._previous_signal_handlers = {}
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
        except (ValueError, OSError):
            self._previous_signal_handlers.clear()

    def _restore_signal_handlers(self) -> None:
        for signum, handler in getattr(self, "_previous_signal_handlers", {}).items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass
        self._previous_signal_handlers = {}

    def _handle_signal(self, signum, frame) -> None:
        self._signal_stop_requested = True
        self._signal_stop_reason = signal.Signals(signum).name
        try:
            self._write_checkpoint(completed=False)
        except Exception:
            self._notify(
                "log",
                "exception",
                "Unable to write Darwin checkpoint after %s",
                self._signal_stop_reason,
                exc_info=True,
            )
