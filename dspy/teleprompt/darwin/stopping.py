"""Composable stopping conditions for Darwin strategies."""

from pathlib import Path
from typing import Any, Protocol


class Stopper(Protocol):
    """Callable protocol evaluated with the active strategy."""

    def __call__(self, strategy: Any) -> bool:
        ...


class ScoreThresholdStopper:
    """Stop once the best observed candidate reaches ``threshold``."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, strategy: Any) -> bool:
        candidate = getattr(strategy, "best_candidate", None)
        return candidate is not None and candidate.average_score() >= self.threshold


class NoImprovementStopper:
    """Stop after ``patience`` generations without improvement."""

    def __init__(self, patience: int):
        if patience < 0:
            raise ValueError("patience must be non-negative")
        self.patience = patience

    def __call__(self, strategy: Any) -> bool:
        return getattr(strategy, "generations_without_improvement", 0) >= self.patience


class FileStopper:
    """Stop when an external control file exists."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __call__(self, strategy: Any) -> bool:
        return self.path.exists()


class AnyStopper:
    """Stop when any supplied stopper requests termination."""

    def __init__(self, *stoppers: Stopper):
        self.stoppers = tuple(stoppers)

    def __call__(self, strategy: Any) -> bool:
        return any(stopper(strategy) for stopper in self.stoppers)
