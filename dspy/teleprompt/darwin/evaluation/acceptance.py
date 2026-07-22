"""Acceptance criteria for GEPA candidate proposals."""

from typing import Iterable


class StrictImprovementAcceptance:
    """Accept a child only when it strictly improves over its parent."""

    def should_accept(self, child_scores: Iterable[float], parent_scores: Iterable[float]) -> bool:
        return sum(child_scores) > sum(parent_scores)


class ImprovementOrEqualAcceptance:
    """Accept children that improve or remain equal to their parent."""

    def should_accept(self, child_scores: Iterable[float], parent_scores: Iterable[float]) -> bool:
        return sum(child_scores) >= sum(parent_scores)
