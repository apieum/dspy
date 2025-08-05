"""Pareto-based candidate selection implementing Algorithm 2."""

import logging
import random
from collections import defaultdict
from typing import List, Dict

from .selection import Selection
from ..data import Candidate, CandidatePool, Cohort
from ..budget import Budget

logger = logging.getLogger(__name__)


class ParetoSelection(Selection):
    """Pareto-based candidate selection (Algorithm 2 from GEPA paper)."""

    def __init__(self):
        self.selection_counts = defaultdict(int)

    def filter(self, pool: CandidatePool, budget: Budget=None) -> Cohort:
        """Select candidates using Pareto-based illumination strategy (Algorithm 2).

        Implementation of Algorithm 2 from GEPA paper:
        1. Get candidates that achieve best score on at least one training task (provided by CandidatePool)
        2. Exclude candidates that have already been used for generation (had_child = True)
        3. Prune strictly dominated candidates
        4. Return Pareto frontier candidates
        """
        self.budget = budget
        
        def pareto_filter(task_scores: Dict[int, Candidate]):
            # Collect all unique candidates from the task winners
            unique_candidates = set(task_scores.values())

            if not unique_candidates:
                return []

            # Exclude candidates that have already been used for generation
            eligible_candidates = [c for c in unique_candidates if not c.had_child]

            if not eligible_candidates:
                return []

            # Apply Pareto selection - remove dominated candidates
            return self._remove_dominated_candidates(eligible_candidates)
        
        candidates = pool.filter_by_task_scores(pareto_filter)
        return Cohort(*candidates)


    def _remove_dominated_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Remove candidates dominated by others using Pareto selection.

        A candidate is dominated if another candidate performs at least as well
        on all tasks and strictly better on at least one task.
        """
        pareto_frontier = []

        for candidate in candidates:
            is_dominated = False

            # Check if this candidate is dominated by any candidate already in frontier
            for frontier_candidate in pareto_frontier:
                if self._dominates(frontier_candidate, candidate):
                    is_dominated = True
                    break

            if not is_dominated:
                # Remove any frontier candidates dominated by this candidate
                pareto_frontier = [fc for fc in pareto_frontier if not self._dominates(candidate, fc)]
                pareto_frontier.append(candidate)

        return pareto_frontier

    def _dominates(self, candidate_a: Candidate, candidate_b: Candidate) -> bool:
        """Check if candidate A dominates candidate B.

        A dominates B if A performs at least as well on all tasks
        and strictly better on at least one task.
        """
        num_tasks = min(len(candidate_a.task_scores), len(candidate_b.task_scores))

        at_least_as_good_on_all = True
        strictly_better_on_one = False

        for task_id in range(num_tasks):
            score_a = candidate_a.task_scores[task_id] if task_id < len(candidate_a.task_scores) else 0.0
            score_b = candidate_b.task_scores[task_id] if task_id < len(candidate_b.task_scores) else 0.0

            if score_a < score_b:
                at_least_as_good_on_all = False
                break

            if score_a > score_b:
                strictly_better_on_one = True

        return at_least_as_good_on_all and strictly_better_on_one
