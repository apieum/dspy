"""Cohort data structure for GEPA optimization."""
import random
import time
from typing import Dict, List, Set, Callable, Any

from ..data.candidate import Candidate

class Cohort:
    """A cohort of candidates created in a single iteration.

    Represents a group of candidates that were generated together
    and share the same iteration context. The iteration is stored
    in the candidates themselves as they're always from the same iteration.
    """

    def __init__(self, *candidates: 'Candidate', iteration: int = -1, **stochastic_weights):
        """Initialize cohort with candidates.

        Args:
            *candidates: Variable number of candidate arguments, or a single list of candidates
            iteration: Iteration number for this cohort (defaults to -1)
            stochastic_weights: kwargs used as weights for sample_stochastic
        """
        # Handle both Cohort(candidate1, candidate2) and Cohort([candidate1, candidate2])
        if len(candidates) == 1 and isinstance(candidates[0], (list, set)):
            self.candidates: Set['Candidate'] = set(candidates[0])
        else:
            self.candidates: Set['Candidate'] = set(candidates)

        self.creation_timestamp: float = time.time()
        self._iteration: int = iteration
        self.weights: Dict[str, Dict[Candidate, float]] = {}
        for name, feature in stochastic_weights.items():
            self.weights[name] = self._extract_feature_values(feature)

    def _extract_feature_values(self, feature_spec: Any) -> Dict[Candidate, float]:
        """Internal helper to resolve a feature specification into a list of float values."""
        if isinstance(feature_spec, dict):
            return {c: float(feature_spec.get(c, 0.0)) for c in self.candidates}
        if callable(feature_spec):
            return {c:float(feature_spec(c)) for c in self.candidates}
        if isinstance(feature_spec, (list, tuple)):
            if len(feature_spec) != len(self.candidates):
                raise ValueError(
                    f"Sequence feature has length {len(feature_spec)} but cohort has "
                    f"{len(self.candidates)} candidates. They must match."
                )
            candidates = list(self.candidates)
            return {candidates[index]: float(feature_spec[index]) for index in range(len(candidates))}
        if isinstance(feature_spec, (int, float)):
            return {c: float(feature_spec) for c in self.candidates}

        raise TypeError(f"Unsupported feature specification type: {type(feature_spec)}")

    def add_feature(self, name:str, values):
        self.weights[name] = self._extract_feature_values(values)

    def del_feature(self, name:str):
        if name in self.weights:
            del self.weights[name]

    @property
    def iteration(self) -> int:
        """Get iteration number for this cohort."""
        return self._iteration

    def __getattr__(self, name: str) -> Any:
        """Get attribute from cohort."""
        if name in self.weights:
            return self.weights[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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

    def sample(self, n: int = 1, exclude: List[Candidate] = [], rng = random) -> 'Cohort':
        """Sample a subset of candidates from this cohort.

        Args:
            n: Number of candidates to sample

        Returns:
            New Cohort of randomly sampled candidates
        """
        candidates = [c for c in self.candidates if c not in exclude]
        candidates = rng.sample(candidates, n)
        new_weights = self._new_weights(candidates)
        return self.__class__(*candidates, iteration=self._iteration, **new_weights)

    def sample_stochastic(self, n: int = 1, exclude: List[Candidate] = [], rng = random, **factors) -> 'Cohort':
        """Stochastic sampling based on task winning frequency (Algorithm 2 line 14).

        Implements GEPA Algorithm 2's stochastic selection and more: "Sample Φk from Ĉ with probability ∝ f[Φk]"
        where f[Φk] is the number of tasks on which candidate Φk achieves the best score.

        Args:
            n: Number of candidates to sample
            **factors: Keyword arguments where keys match feature names provided during
                cohort initialization (e.g., `task_wins=2.0`, `average_score=1.0`).
                The values are float multipliers for each feature.

        Returns:
            Parents cohort with stochastically selected candidates
        """

        candidates = [c for c in self.candidates if c not in exclude]
        pool_size = len(candidates)
        if pool_size <= n:
            new_weights = self._new_weights(candidates)
            return self.__class__(*candidates, iteration=self._iteration, **new_weights)


        # If no explicit factors are given, use all features from __init__ with weight 1.0.
        factors = {key: factors.get(key, 1.0) for key in self.weights}

        abs = lambda val: (val >= 0.0 and float(val) or 0.0)
        final_weights = [0.0] * pool_size
        for feature, weights in self.weights.items():
            final_weights = [final_weights[i] + abs(weights[candidates[i]] * factors[feature]) for i in range(pool_size)]

        # Handle case where all weights are 0
        if sum(final_weights) == 0:
            final_weights = [1.0] * pool_size

        # Sample n candidates based on weights
        selected = rng.choices(candidates, weights=final_weights, k=n)
        new_weights = self._new_weights(selected)
        return self.__class__(*selected, iteration=self._iteration, **new_weights)

    def _new_weights(self, candidates: List[Candidate] = []):
        new_weights = {}
        for feature, weights in self.weights.items():
             new_weights[feature] = {candidate: float(weights.get(candidate, 0.0)) for candidate in candidates}
        return new_weights


    def filter(self, comparison_func:Callable) -> 'Cohort':
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

        new_weights = self._new_weights(kept_candidates)
        return self.__class__(*kept_candidates, iteration=self._iteration, **new_weights)



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

    def __init__(self, *candidates: 'Candidate', **kwargs):
        """Initialize Parents cohort with optional task win counts.

        Args:
            *candidates: Candidate objects to include
            **kwargs: Additional arguments passed to Cohort
        """
        super().__init__(*candidates, **kwargs)

class NewBorns(Cohort):
    """Cohort of newly generated candidates.

    These candidates have been created through mutation or merge
    and need to be evaluated before they can become survivors.
    """

    def __init__(self, *candidates: 'Candidate', **kwargs):
        super().__init__(*candidates, **kwargs)
