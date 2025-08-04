"""Evaluator protocol for GEPA optimization."""

from abc import abstractmethod
from typing import Callable, List, Protocol, TYPE_CHECKING
import dspy
from ..compilation_observer import CompilationObserver

if TYPE_CHECKING:
    from ..data.cohort import Cohort
    from ..data.candidate_pool import CandidatePool
    from ..budget import Budget


class Evaluator(CompilationObserver):
    """Protocol for evaluating and filtering new candidates.

    This component owns a metric and decides which newly generated
    candidates should be promoted (kept) vs discarded.
    """

    @abstractmethod
    def evaluate(self, cohort: "Cohort", budget: "Budget") -> "Cohort":
        """Evaluate new candidates and filter based on promotion criteria.

        Args:
            cohort: Newly generated candidates to evaluate
            budget: Budget to track costs

        Returns:
            Cohort containing only promoted (worthy) candidates
        """
        ...

    @abstractmethod
    def evaluate_for_promotion(self, cohort: "Cohort", budget: "Budget") -> "Cohort":
        """Evaluate candidates for promotion with budget tracking.

        Args:
            cohort: Candidates to evaluate
            budget: Budget to track costs

        Returns:
            Cohort with evaluated candidates (task_scores populated)
        """
        ...

    @abstractmethod
    def get_metric(self) -> Callable:
        """Get the metric function used by this component."""
        ...
