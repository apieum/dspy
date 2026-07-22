"""Diversity-preserving candidate archive inspired by LEVI-style search."""

from dspy.teleprompt.utils import get_signature


class DiversityArchive:
    """Keep the strongest candidate for each distinct prompt fingerprint."""

    def __init__(self, capacity: int = 32):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._entries = {}

    @staticmethod
    def fingerprint(candidate) -> tuple[str, ...]:
        return tuple(
            f"{get_signature(predictor).signature}\n{get_signature(predictor).instructions}"
            for predictor in candidate.module.predictors()
        )

    def add(self, candidate) -> None:
        key = self.fingerprint(candidate)
        current = self._entries.get(key)
        if current is None or candidate.average_score() > current.average_score():
            self._entries[key] = candidate
        self._trim()

    def add_all(self, candidates) -> None:
        for candidate in candidates:
            self.add(candidate)

    def _trim(self) -> None:
        if len(self._entries) > self.capacity:
            ranked = sorted(self._entries.items(), key=lambda item: item[1].average_score(), reverse=True)
            self._entries = dict(ranked[: self.capacity])

    def candidates(self):
        return list(self._entries.values())

    def __len__(self):
        return len(self._entries)
