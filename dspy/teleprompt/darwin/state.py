"""Serializable optimization checkpoint state."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OptimizationCheckpoint:
    """JSON-safe progress snapshot for a Darwin compilation run."""

    schema_version: int = 1
    generation: int = 0
    algorithm_state: str = "initialize"
    history: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OptimizationCheckpoint":
        return cls(**value)
