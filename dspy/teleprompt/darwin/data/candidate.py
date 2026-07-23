"""Generic candidate model used by Darwin algorithms."""

from dataclasses import dataclass, field
from typing import Any, Generic, List, Protocol, TypeVar


CandidateValue = TypeVar("CandidateValue")


class CandidateOperations(Protocol[CandidateValue]):
    """Operations that an algorithm may inject into a candidate.

    Darwin's data model does not decide how a candidate is evaluated or
    compared.  An algorithm supplies those decisions through this protocol.
    The protocol deliberately leaves the operation arguments open because
    different algorithms evaluate candidates with different contexts.
    """

    def evaluate(self, candidate: "Candidate[CandidateValue]", *args: Any, **kwargs: Any) -> Any:
        ...

    def compare(
        self,
        left: "Candidate[CandidateValue]",
        right: "Candidate[CandidateValue]",
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Return a positive, zero, or negative comparison result."""
        ...


@dataclass(init=False)
class Candidate(Generic[CandidateValue]):
    """A generic value evolving through an algorithm.

    The candidate owns identity, lineage, generation metadata, and an
    optional operations object.  Evaluation, ranking, dominance, and other
    algorithmic decisions are delegated to that object; no GEPA or DSPy
    policy is embedded here.
    """

    value: CandidateValue
    parents: List["Candidate[CandidateValue]"] = field(default_factory=list)
    generation_number: int = 0
    creation_metadata: dict[str, Any] = field(default_factory=dict)
    operations: CandidateOperations[CandidateValue] | None = field(
        default=None, repr=False, compare=False
    )

    def __init__(
        self,
        value: CandidateValue,
        parents: List["Candidate[CandidateValue]"] | None = None,
        generation_number: int = 0,
        creation_metadata: dict[str, Any] | None = None,
        operations: CandidateOperations[CandidateValue] | None = None,
    ) -> None:
        self.value = value
        self.parents = list(parents or [])
        self.generation_number = generation_number
        self.creation_metadata = dict(creation_metadata or {})
        self.operations = operations

    def __hash__(self) -> int:
        """Candidates are identity objects, not value-equal records."""
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate evaluation to the injected algorithm operations."""
        if self.operations is None:
            raise TypeError("candidate operations are required for evaluation")
        return self.operations.evaluate(self, *args, **kwargs)

    def compare(self, other: "Candidate[CandidateValue]", *args: Any, **kwargs: Any) -> int:
        """Delegate comparison to the injected algorithm operations."""
        if self.operations is None:
            raise TypeError("candidate operations are required for comparison")
        return self.operations.compare(self, other, *args, **kwargs)

    def _get_all_ancestors(self) -> set["Candidate[CandidateValue]"]:
        """Return all unique ancestors without imposing algorithm policy."""
        ancestors: set[Candidate[CandidateValue]] = set()
        to_visit = list(self.parents)
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(parent.parents)
        return ancestors

    def is_descendant_of(self, other: "Candidate[CandidateValue]") -> bool:
        return other in self.parents or any(
            parent.is_descendant_of(other) for parent in self.parents
        )

    def is_ancestor_of(self, other: "Candidate[CandidateValue]") -> bool:
        return other.is_descendant_of(self)

    def filter_ancestors(
        self, allowed_ancestors: set["Candidate[CandidateValue]"]
    ) -> set["Candidate[CandidateValue]"]:
        return self._get_all_ancestors().intersection(allowed_ancestors)

    def find_common_ancestors(
        self, other: "Candidate[CandidateValue]"
    ) -> set["Candidate[CandidateValue]"]:
        return other.filter_ancestors(self._get_all_ancestors())

    def is_ancestor_of_any(
        self, candidates: List["Candidate[CandidateValue]"]
    ) -> bool:
        return any(candidate.is_descendant_of(self) for candidate in candidates)
