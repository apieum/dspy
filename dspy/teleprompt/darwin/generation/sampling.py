"""Proposal sampling strategies for Darwin generation steps.

The default strategy preserves the original GEPA behavior: one proposal from
the currently selected parent cohort.  The other strategies are useful when a
generation should explore several mutations before promotion.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence, Any

if TYPE_CHECKING:
    from ..data.cohort import Parents


@dataclass
class ProposalTask:
    """Explicit work item sent through Darwin's proposal pipeline."""

    parents: "Parents"
    feedback_data: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SamplingStrategy(ABC):
    """Select parent cohorts for one generation's proposal tasks."""

    @abstractmethod
    def sample(
        self,
        parents: "Parents",
        count: int = 1,
        *,
        rng: random.Random | None = None,
    ) -> list["Parents"]:
        """Return the parent cohort used by each proposal task."""


class BatchSampler(ABC):
    """Select reflection examples for proposal tasks."""

    @abstractmethod
    def sample(
        self,
        data: Sequence[Any],
        count: int,
        *,
        rng: random.Random | None = None,
    ) -> list[list[Any]]:
        """Return one minibatch for each proposal task."""


class EpochShuffledBatchSampler(BatchSampler):
    """Yield sequential chunks from a shuffled, repeatedly cycling dataset."""

    def __init__(self, minibatch_size: int):
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        self.minibatch_size = minibatch_size
        self._order: list[Any] = []
        self._position = 0

    def sample(self, data, count, *, rng=None):
        if count <= 0:
            raise ValueError("count must be positive")
        if not data:
            return [[] for _ in range(count)]
        rng = rng or random.Random()
        if len(self._order) != len(data) or self._position >= len(self._order):
            self._order = list(data)
            rng.shuffle(self._order)
            self._position = 0

        batches = []
        for _ in range(count):
            if self._position + self.minibatch_size > len(self._order):
                remainder = self._order[self._position:]
                rng.shuffle(self._order)
                self._position = 0
                self._order = self._order
                needed = self.minibatch_size - len(remainder)
                batch = remainder + self._order[:needed]
                self._position = needed
            else:
                batch = self._order[self._position:self._position + self.minibatch_size]
                self._position += self.minibatch_size
            batches.append(list(batch))
        return batches


class SingleMutationSampling(SamplingStrategy):
    """One proposal, matching classic GEPA behavior.

    ``count`` is honored for compatibility with Darwin's existing
    ``proposals_per_generation`` setting.  With its default value of one this
    is exactly one parent/mutation task.
    """

    def sample(self, parents, count=1, *, rng=None):
        if count <= 0:
            raise ValueError("count must be positive")
        return [parents] * count


class SameParentSampling(SamplingStrategy):
    """Generate N proposals from one parent, using separate mutations."""

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n

    def sample(self, parents, count=1, *, rng=None):
        if parents.is_empty():
            return []
        rng = rng or random.Random()
        selected = parents.sample_stochastic(1, rng=rng)
        return [selected] * self.n


class IndependentSampling(SamplingStrategy):
    """Generate N proposals, independently selecting a parent each time."""

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n

    def sample(self, parents, count=1, *, rng=None):
        if parents.is_empty():
            return []
        rng = rng or random.Random()
        return [parents.sample_stochastic(1, rng=rng) for _ in range(self.n)]


class PxNSampling(SamplingStrategy):
    """Select P parents and generate N proposals from each: P × N tasks."""

    def __init__(self, p: int, n: int):
        if p <= 0 or n <= 0:
            raise ValueError("p and n must be positive")
        self.p = p
        self.n = n

    def sample(self, parents, count=1, *, rng=None):
        if parents.is_empty():
            return []
        rng = rng or random.Random()
        tasks = []
        for _ in range(self.p):
            selected = parents.sample_stochastic(1, rng=rng)
            tasks.extend([selected] * self.n)
        return tasks


__all__ = [
    "ProposalTask",
    "SamplingStrategy",
    "BatchSampler",
    "EpochShuffledBatchSampler",
    "SingleMutationSampling",
    "SameParentSampling",
    "IndependentSampling",
    "PxNSampling",
]
