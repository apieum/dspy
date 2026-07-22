"""Factories for the role-specific cohorts used by a Darwin strategy.

The cohort roles are part of an optimization model, not a requirement of the
checkpoint or evaluator implementations.  A strategy may therefore provide
its own cohort classes (or factories) while the default GEPA recipe keeps the
familiar ``Parents``, ``NewBorns`` and ``Survivors`` roles.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .cohort import Cohort, NewBorns, Parents, Survivors


@dataclass(frozen=True)
class CohortModel:
    """Create the cohort roles required by an optimization strategy.

    The model deliberately contains no selection or restoration logic.  It
    only defines how a strategy materializes its domain cohorts.  This keeps
    selection policies free to choose *which* candidates belong in a cohort,
    while allowing another strategy to use different cohort implementations.
    """

    cohort_type: type[Cohort] = Cohort
    parents_type: type[Cohort] = Parents
    newborns_type: type[Cohort] = NewBorns
    survivors_type: type[Cohort] = Survivors

    def _factory(self, role: str):
        try:
            return {
                "cohort": self.cohort_type,
                "parents": self.parents_type,
                "newborns": self.newborns_type,
                "survivors": self.survivors_type,
            }[role]
        except KeyError as exc:
            raise ValueError(f"Unknown cohort role: {role}") from exc

    def create(
        self,
        role: str,
        candidates: Iterable[Any] = (),
        *,
        iteration: int = -1,
        **features: Mapping[Any, float] | float,
    ) -> Cohort:
        """Materialize a cohort for ``role`` with its feature metadata."""
        factory = self._factory(role)
        return factory(*list(candidates), iteration=iteration, **features)

    def cohort(self, candidates=(), *, iteration: int = -1, **features) -> Cohort:
        return self.create("cohort", candidates, iteration=iteration, **features)

    def parents(self, candidates=(), *, iteration: int = -1, **features) -> Cohort:
        return self.create("parents", candidates, iteration=iteration, **features)

    def newborns(self, candidates=(), *, iteration: int = -1, **features) -> Cohort:
        return self.create("newborns", candidates, iteration=iteration, **features)

    def survivors(self, candidates=(), *, iteration: int = -1, **features) -> Cohort:
        return self.create("survivors", candidates, iteration=iteration, **features)

