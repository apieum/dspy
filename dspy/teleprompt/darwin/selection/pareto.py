"""Base Pareto selection logic for multi-objective optimization.

Provides core Pareto dominance filtering logic that can be used
by different selection strategies.
"""

from typing import Set, TypeVar, Protocol, Callable
from ..data.candidate import Candidate


T = TypeVar('T')


class Dominates(Protocol):
    """Protocol for candidates that support dominance comparison."""

    def dominate(self, other: "Dominates") -> bool:
        """Check if this candidate dominates another.

        Args:
            other: Other candidate to compare against

        Returns:
            True if this candidate dominates other
        """
        ...


def remove_dominated(candidates: Set[T], dominance_fn: Callable[[T, T], bool]) -> Set[T]:
    """Remove dominated candidates from a set.

    A candidate is dominated if there exists another candidate that is
    better or equal on all objectives and strictly better on at least one.

    Args:
        candidates: Set of candidates to filter
        dominance_fn: Function (candidate1, candidate2) -> bool
                     Returns True if candidate1 dominates candidate2

    Returns:
        Set of non-dominated candidates (Pareto frontier)
    """
    candidates = set(candidates)

    return {
        candidate
        for candidate in candidates
        if not any(
            other is not candidate and dominance_fn(other, candidate)
            for other in candidates
        )
    }


def remove_dominated_with_method(candidates: Set[Dominates]) -> Set[Dominates]:
    """Remove dominated candidates using their .dominate() method.

    Convenience function for candidates that implement the Dominates protocol.

    Args:
        candidates: Set of candidates with .dominate() method

    Returns:
        Set of non-dominated candidates (Pareto frontier)
    """
    return remove_dominated(candidates, lambda a, b: a.dominate(b))


class ParetoMixin:
    """Mixin providing Pareto filtering utilities for selectors.

    Can be mixed into any selector that needs Pareto dominance logic.

    Example:
        class MySelector(Selector, ParetoMixin):
            def promote(self, survivors, budget):
                # Get candidates
                all_candidates = self._get_all_candidates()

                # Filter to Pareto frontier
                frontier = self.filter_pareto(all_candidates)

                # Return parents
                return Parents(*frontier, iteration=survivors.iteration + 1)
    """

    @staticmethod
    def filter_pareto(
        candidates: Set[T],
        dominance_fn: Callable[[T, T], bool] = None
    ) -> Set[T]:
        """Filter candidates to Pareto frontier.

        Args:
            candidates: Set of candidates to filter
            dominance_fn: Optional dominance function. If None, uses candidate.dominate()

        Returns:
            Set of non-dominated candidates
        """
        if dominance_fn is None:
            return remove_dominated_with_method(candidates)
        else:
            return remove_dominated(candidates, dominance_fn)

    @staticmethod
    def is_dominated(candidate: T, population: Set[T], dominance_fn: Callable[[T, T], bool] = None) -> bool:
        """Check if a candidate is dominated by any member of population.

        Args:
            candidate: Candidate to check
            population: Population to compare against
            dominance_fn: Optional dominance function. If None, uses dominate() method

        Returns:
            True if candidate is dominated by any member of population
        """
        if dominance_fn is None:
            return any(
                other is not candidate and other.dominate(candidate)
                for other in population
            )
        else:
            return any(
                other is not candidate and dominance_fn(other, candidate)
                for other in population
            )
