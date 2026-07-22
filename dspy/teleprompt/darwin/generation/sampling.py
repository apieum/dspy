"""Proposal sampling strategies for Darwin generation steps.

The default strategy preserves the original GEPA behavior: one proposal from
the currently selected parent cohort.  The other strategies are useful when a
generation should explore several mutations before promotion.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.cohort import Parents


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
    "SamplingStrategy",
    "SingleMutationSampling",
    "SameParentSampling",
    "IndependentSampling",
    "PxNSampling",
]
