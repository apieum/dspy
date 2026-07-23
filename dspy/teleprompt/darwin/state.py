"""Serializable optimization checkpoint state."""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for objects whose runtime state belongs in a checkpoint.

    ``serialize_state`` must return JSON-compatible data and must not perform
    file I/O. ``restore_state`` mutates an already-configured object; it does
    not reconstruct configuration or external resources. The enclosing
    ``OptimizationCheckpoint`` owns schema versioning for the payload.
    """

    def serialize_state(self) -> dict[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...


@dataclass
class OptimizationCheckpoint:
    """JSON-safe progress snapshot for a Darwin compilation run."""

    schema_version: int = 1
    generation: int = 0
    algorithm_state: str = "initialize"
    history: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    strategy_state: dict[str, Any] = field(default_factory=dict)
    rng_state: Any = None
    stop_reason: str | None = None
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OptimizationCheckpoint":
        if not isinstance(value, dict):
            raise TypeError("checkpoint payload must be a dictionary")

        schema_version = int(value.get("schema_version", 1))
        if schema_version > cls.schema_version:
            raise ValueError(
                f"unsupported checkpoint schema version {schema_version}; "
                f"this runtime supports up to {cls.schema_version}"
            )

        # Ignore fields added by newer compatible writers.  All fields added
        # to the current schema have defaults, so older checkpoints remain
        # loadable without special-case migrations.
        known_fields = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in known_fields}
        payload["schema_version"] = schema_version
        return cls(**payload)
