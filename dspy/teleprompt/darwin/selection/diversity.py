"""Diversity-preserving candidate archive for maintaining unique solutions."""

from typing import Callable, Generic, TypeVar, Dict, List, Any, Optional


T = TypeVar('T')


class DiversityArchive(Generic[T]):
    """Keep the strongest candidate for each distinct fingerprint.

    Maintains diversity by storing only the best candidate for each unique
    fingerprint, preventing redundant similar solutions.

    Attributes:
        capacity: Maximum number of unique fingerprints to store
        fingerprint_fn: Function to compute fingerprint from candidate
        score_fn: Function to compute score from candidate for comparison
    """

    def __init__(
        self,
        capacity: int = 32,
        fingerprint_fn: Optional[Callable[[T], Any]] = None,
        score_fn: Optional[Callable[[T], float]] = None,
    ):
        """Initialize diversity archive.

        Args:
            capacity: Maximum number of unique fingerprints to store
            fingerprint_fn: Function mapping candidate to hashable fingerprint
            score_fn: Function mapping candidate to numeric score

        Raises:
            ValueError: If capacity is not positive
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.fingerprint_fn = fingerprint_fn or self._default_fingerprint
        self.score_fn = score_fn or self._default_score
        self._entries: Dict[Any, T] = {}

    @staticmethod
    def _default_fingerprint(candidate: T) -> Any:
        """Default fingerprint using object id."""
        return id(candidate)

    @staticmethod
    def _default_score(candidate: T) -> float:
        """Default score returns 0.0."""
        return 0.0

    def add(self, candidate: T) -> None:
        """Add candidate to archive, replacing if better than existing.

        Args:
            candidate: Candidate to add
        """
        key = self.fingerprint_fn(candidate)
        current = self._entries.get(key)

        if current is None or self.score_fn(candidate) > self.score_fn(current):
            self._entries[key] = candidate

        self._trim()

    def add_all(self, candidates: List[T]) -> None:
        """Add multiple candidates to archive.

        Args:
            candidates: List of candidates to add
        """
        for candidate in candidates:
            self.add(candidate)

    def _trim(self) -> None:
        """Trim archive to capacity, keeping highest scoring entries."""
        if len(self._entries) > self.capacity:
            ranked = sorted(
                self._entries.items(),
                key=lambda item: self.score_fn(item[1]),
                reverse=True
            )
            self._entries = dict(ranked[:self.capacity])

    def candidates(self) -> List[T]:
        """Get all candidates in archive.

        Returns:
            List of candidates
        """
        return list(self._entries.values())

    def __len__(self) -> int:
        """Return number of unique fingerprints in archive."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all entries from archive."""
        self._entries.clear()
