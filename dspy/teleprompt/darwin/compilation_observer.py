"""Lifecycle observer protocol for Darwin compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import dspy

if TYPE_CHECKING:
    from .budget import Budget
    from .data.cohort import Cohort
    from .dataset_manager import DatasetManager


class CompilationObserver(Protocol):
    """Optional callbacks for compilation and iteration lifecycle events."""

    def start_compilation(self, student: dspy.Module, dataset_manager: "DatasetManager") -> None: ...

    def finish_compilation(self, result: dspy.Module) -> None: ...

    def start_iteration(self, iteration: int, cohort: "Cohort", budget: "Budget") -> None: ...

    def finish_iteration(self, iteration: int, cohort: "Cohort", budget: "Budget") -> None: ...

    def candidate_evaluated(self, candidate, accepted: bool) -> None: ...

    def budget_exhausted(self, budget: "Budget") -> None: ...
