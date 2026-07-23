"""Budget contract for Darwin compilation runs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol
from ..state import Checkpointable


class BudgetStrategy(Protocol):
    """Minimal strategy surface a budget may use during reconciliation."""

    def request_stop(self, reason: str) -> None: ...


class BudgetExhaustedError(RuntimeError):
    """Raised when a budget cannot admit a requested operation."""


@dataclass(frozen=True)
class BudgetEvent:
    """Description of one operation that may consume budget."""

    phase: str
    units: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Budget(Checkpointable, ABC):
    """Manage resource limits and their lifecycle for one compilation run.

    Components may report or reserve work through the cost methods below. The
    strategy never inspects the budget's counters: at each lifecycle boundary
    it asks the budget to reconcile itself, and the budget requests a stop when
    its policy can no longer allow useful work.
    """

    @abstractmethod
    def reconcile(self, strategy: BudgetStrategy) -> None:
        """Enforce the budget policy and request a strategy stop if needed."""
        ...

    @abstractmethod
    def spend(self, event: BudgetEvent) -> None:
        """Admit and account for one operation, or raise if it cannot run."""
        ...

    @abstractmethod
    def serialize_state(self) -> dict[str, Any]:
        """Return JSON-safe accounting state for a checkpoint."""
        ...

    @classmethod
    @abstractmethod
    def restore_state(cls, state: Mapping[str, Any]):
        """Construct a new budget entirely from serialized accounting state."""
        ...


def configured_limit(
    explicit: int | None,
    config: object | None,
    attribute: str,
    description: str,
) -> int:
    """Resolve a constructor limit while keeping restoration independent.

    Direct component construction may provide a limit explicitly; strategy
    construction normally supplies it through the current configuration.
    This helper is deliberately limited to construction-time defaults. It is
    not used by checkpoint restoration, where the serialized state is the
    complete source of truth.
    """
    value = explicit if explicit is not None else getattr(config, attribute, None)
    if value is None:
        raise TypeError(description)
    return int(value)
