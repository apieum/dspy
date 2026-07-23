"""Budget contract for Darwin compilation runs."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol, TYPE_CHECKING
import dspy

if TYPE_CHECKING:
    from ..config import DarwinConfig


class BudgetStrategy(Protocol):
    """Minimal strategy surface a budget may use during reconciliation."""

    def request_stop(self, reason: str) -> None: ...


class Budget(ABC):
    """Manage resource limits and their lifecycle for one compilation run.

    Components may report or reserve work through the cost methods below. The
    strategy never inspects the budget's counters: at each lifecycle boundary
    it asks the budget to reconcile itself, and the budget requests a stop when
    its policy can no longer allow useful work.
    """

    def reset(self) -> None:
        """Reset accounting before a new compilation run."""

    def reconcile(self, strategy: BudgetStrategy) -> None:
        """Ask the budget to enforce its stopping policy on ``strategy``."""
        if self.is_exhausted():
            strategy.request_stop("budget_exhausted")

    @abstractmethod
    def is_exhausted(self) -> bool:
        """Return whether this budget can no longer allow useful work."""
        ...

    def can_spend(self, phase: str, units: int = 1) -> bool:
        """Return whether a component may reserve ``units`` of work.

        Budgets without phase-specific accounting use exhaustion as their
        policy. Concrete budgets with domains should override this method.
        """
        if units < 0:
            raise ValueError("units must be non-negative")
        return not self.is_exhausted()

    def start_iteration(self, iteration: int, cohort=None) -> None:
        """Observe the beginning of an algorithm iteration, if relevant."""

    def finish_iteration(self, iteration: int, cohort=None) -> None:
        """Observe the end of an algorithm iteration, if relevant."""

    def serialize_state(self) -> dict[str, Any]:
        """Return JSON-safe accounting state for a checkpoint."""
        return dict(self.get_remaining())

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore accounting state from a checkpoint."""
        del state

    def __float__(self) -> float:
        """Convert budget to float (remaining budget value)."""
        remaining = self.get_remaining()
        if isinstance(remaining, dict):
            # Get the primary budget value (first key)
            primary_key = next(iter(remaining.keys()))
            return float(remaining[primary_key])
        return float(remaining)

    def __int__(self) -> int:
        """Convert budget to int (remaining budget value)."""
        remaining = self.get_remaining()
        if isinstance(remaining, dict):
            # Get the primary budget value (first key)
            primary_key = next(iter(remaining.keys()))
            return int(remaining[primary_key])
        return int(remaining)

    def __gt__(self, other) -> bool:
        """Magic comparison for `budget > value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) > other
        return NotImplemented

    def __lt__(self, other) -> bool:
        """Magic comparison for `budget < value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) < other
        return NotImplemented

    def __le__(self, other) -> bool:
        """Magic comparison for `budget <= value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) <= other
        return NotImplemented

    def __ge__(self, other) -> bool:
        """Magic comparison for `budget >= value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) >= other
        return NotImplemented

    def __eq__(self, other) -> bool:
        """Magic comparison for `budget == value`."""
        if isinstance(other, (int, float)):
            return type(other)(self) == other
        return NotImplemented

    def __ne__(self, other) -> bool:
        """Magic comparison for `budget != value`."""
        return not self.__eq__(other)

    def spend_on_evaluation(self, module: dspy.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of evaluating a candidate module (LLM calls).

        Args:
            module: DSPy module that was evaluated
            metadata: Optional details like {"phase": "minibatch", "examples": 3}
        """
        pass  # Override in child classes if needed

    def spend_on_generation(self, module: Optional[dspy.Module] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of generating new candidates (LLM calls for reflection/mutation).

        Args:
            module: Optional module being mutated/generated from
            metadata: Optional details like {"type": "reflection", "strategy": "mutation"}
        """
        pass  # Override in child classes if needed

    def spend_on_selection(self, candidates_to_promote: int, candidates_selected: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track cost of candidate selection (usually algorithmic, no LLM calls).

        Args:
            candidates_to_promote: Total candidates available for selection
            candidates_selected: Number of candidates actually selected
            metadata: Optional details like {"strategy": "pareto", "tasks": 150}
        """
        pass  # Override in child classes if needed

    def get_remaining(self) -> dict:
        """Return a human-readable remaining budget breakdown."""
        raise NotImplementedError
