"""Base interface for candidate selection strategies."""

from abc import ABC, abstractmethod
from typing import List

import dspy

from ..data.structures import ScoreMatrix


class CandidateSelector(ABC):
    """Interface for candidate selection strategies (Algorithm 2 from paper)."""

    @abstractmethod
    def select_candidate(self, candidates: List[dspy.Module], scores: ScoreMatrix) -> int:
        """Select candidate index using selection strategy."""
        raise NotImplementedError