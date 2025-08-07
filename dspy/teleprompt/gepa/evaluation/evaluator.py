"""Evaluator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import Callable, TYPE_CHECKING
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data.cohort import Survivors, NewBorns
    from ..budget import Budget


class Evaluator(CompilationObserver):
    """Protocol for evaluating and filtering new candidates.

    This component owns a metric and decides which newly generated
    candidates should be promoted (kept) vs discarded. It encapsulates
    the two-phase evaluation logic from the GEPA paper.
    """

    @abstractmethod
    def evaluate(self, cohort: "NewBorns", budget: "Budget") -> "Survivors":
        """
        Evaluates new candidates. If a candidate has parents, it undergoes
        two-phase validation. If it has no parents (the initial candidate),
        it is automatically promoted to full evaluation.

        Args:
            cohort: The cohort of newly generated candidates to evaluate.
            budget: The budget manager to track evaluation costs.

        Returns:
            A Survivors cohort containing only the candidates that were
            successfully promoted after passing evaluation.
        """
        ...

    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the metric function used by this component."""
        ...
