"""Compilation-scoped cache for candidate/example evaluations."""

from typing import Any


class EvaluationCache:
    """Cache metric results by candidate identity and example identity.

    The cache is intentionally scoped to one compilation. Candidate modules
    are immutable optimization artifacts during a run, so object identity is
    the safest key and avoids accidentally reusing results across runs.
    """

    def __init__(self):
        self._values: dict[tuple[int, int], Any] = {}

    @staticmethod
    def _example_key(example):
        return getattr(example, "dspy_uuid", None) or id(example)

    def get(self, candidate, example):
        return self._values.get((id(candidate), self._example_key(example)))

    def put(self, candidate, example, value) -> None:
        self._values[(id(candidate), self._example_key(example))] = value

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)
