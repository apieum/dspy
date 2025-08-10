"""Cohort data structure for GEPA optimization."""
import random
import time
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

# Import Candidate with relative import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..data.candidate import Candidate


class Cohort:
    """A cohort of candidates created in a single iteration.

    Represents a group of candidates that were generated together
    and share the same iteration context. The iteration is stored
    in the candidates themselves as they're always from the same iteration.
    """

    def __init__(self, *candidates: 'Candidate', creation_timestamp: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None, iteration: int = -1):
        """Initialize cohort with candidates.

        Args:
            *candidates: Variable number of candidate arguments, or a single list of candidates
            creation_timestamp: When cohort was created (defaults to current time)
            metadata: Optional metadata dict
            iteration: Iteration number for this cohort (defaults to -1)
        """
        # Handle both Cohort(candidate1, candidate2) and Cohort([candidate1, candidate2])
        if len(candidates) == 1 and isinstance(candidates[0], (list, set)):
            self.candidates: Set['Candidate'] = set(candidates[0])
        else:
            self.candidates: Set['Candidate'] = set(candidates)

        self.creation_timestamp: float = creation_timestamp or time.time()
        self.metadata: Dict[str, Any] = metadata or {}
        self._iteration: int = iteration

    @property
    def iteration(self) -> int:
        """Get iteration number for this cohort."""
        return self._iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        """Set iteration number for this cohort."""
        self._iteration = value

    def is_empty(self) -> bool:
        """Check if cohort has no candidates."""
        return len(self.candidates) == 0

    def size(self) -> int:
        """Number of candidates in this cohort."""
        return len(self.candidates)

    def __iter__(self):
        """Make cohort iterable over candidates."""
        return iter(self.candidates)

    def __len__(self):
        """Return number of candidates."""
        return len(self.candidates)

    def contains(self, candidate: 'Candidate') -> bool:
        """Check if cohort contains a specific candidate."""
        return candidate in self.candidates

    def to_list(self) -> List['Candidate']:
        """Get all candidates as a list (for cases where order doesn't matter)."""
        return list(self.candidates)

    def first(self) -> 'Candidate':
        """Get the first candidate from this cohort.

        Returns:
            First candidate from the set

        Raises:
            StopIteration: If cohort is empty
        """
        return next(iter(self.candidates))

    def sample(self, size: int) -> List['Candidate']:
        """Sample a subset of candidates from this cohort.

        Args:
            size: Number of candidates to sample

        Returns:
            List of sampled candidates
        """
        return random.sample(list(self.candidates), size)

    def filter(self, comparison_func) -> 'Cohort':
        """Filter candidates using a comparison function.

        Args:
            comparison_func: Function that takes (candidate1, candidate2) -> bool
                           Returns True if candidate1 should exclude candidate2

        Returns:
            New Cohort with filtered candidates
        """
        if self.is_empty():
            return Cohort()

        kept_candidates = []
        candidates_copy = list(self.candidates)

        while candidates_copy:
            # Pop one candidate to test
            candidate = candidates_copy.pop(0)

            # Check if any remaining candidate excludes this one
            is_excluded = any(comparison_func(other, candidate) for other in candidates_copy)

            if not is_excluded:
                kept_candidates.append(candidate)

        return Cohort(*kept_candidates)



class Survivors(Cohort):
    """Cohort of candidates that survived Pareto selection.

    These candidates represent the Pareto frontier and are eligible
    for selection as parents in the next generation.
    """

    def __init__(self, *candidates: 'Candidate', **kwargs):
        super().__init__(*candidates, **kwargs)


class Parents(Cohort):
    """Cohort of candidates selected for reproduction.

    These candidates have been chosen through stochastic selection
    and will be used for mutation or merge operations.
    """

    def __init__(self, *candidates: 'Candidate', task_wins: Optional[Dict['Candidate', int]] = None, **kwargs):
        """Initialize Parents cohort with optional task win counts.

        Args:
            *candidates: Candidate objects to include
            task_wins: Optional pre-computed mapping of candidate -> number of tasks won
                      If provided, avoids recalculation in sample_stochastic()
            **kwargs: Additional arguments passed to Cohort
        """
        super().__init__(*candidates, **kwargs)
        self.task_wins = task_wins or {}

    def sample_stochastic(self, n: int = 1) -> 'Parents':
        """Stochastic sampling based on task winning frequency (Algorithm 2 line 14).

        Implements Algorithm 2's stochastic selection: "Sample Φk from Ĉ with probability ∝ f[Φk]"
        where f[Φk] is the number of tasks on which candidate Φk achieves the best score.

        Args:
            n: Number of candidates to sample

        Returns:
            Parents cohort with stochastically selected candidates
        """
        if self.is_empty():
            return Parents()

        if len(self.candidates) <= n:
            return Parents(*self.candidates, task_wins=self.task_wins)

        # Use pre-computed task_wins if available, otherwise calculate
        if not self.task_wins:
            raise ValueError(
                "sample_stochastic requires pre-computed task_wins. "
                "Please provide task_wins when initializing the Parents cohort."
            )
            return

        # Create weights based on task wins (default to 0 if not in dict)
        weights = [self.task_wins.get(candidate, 0) for candidate in self]

        # Handle case where all weights are 0
        if sum(weights) == 0:
            weights = [1] * self.size()

        # Sample one candidate based on weights
        selected = random.choices(list(self), weights=weights, k=n)
        task_wins = {candidate: self.task_wins.get(candidate, 0) for candidate in selected}

        return Parents(*selected, task_wins=task_wins)


class NewBorns(Cohort):
    """Cohort of newly generated candidates.

    These candidates have been created through mutation or merge
    and need to be evaluated before they can become survivors.
    """

    def __init__(self, *candidates: 'Candidate', **kwargs):
        super().__init__(*candidates, **kwargs)
