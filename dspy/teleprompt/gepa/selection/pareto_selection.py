"""Pareto-based candidate selection implementing Algorithm 2."""

import logging
import random
from collections import defaultdict
from typing import List, Dict

from .selection import Selection
from ..data import Candidate, CandidatePool
from ..budget import Budget

logger = logging.getLogger(__name__)


class ParetoSelection(Selection):
    """Pareto-based candidate selection (Algorithm 2 from GEPA paper)."""

    def __init__(self):
        self.selection_counts = defaultdict(int)

    def filter(self, pool: CandidatePool, budget: Budget=None):
        self.budget = budget
        return pool.filter_by_task_scores(self.filter_scores)

    def filter_scores(self, task_scores: Dict[int, Candidate]) -> List[Candidate]:
        """Select candidates using Pareto-based illumination strategy (Algorithm 2).

        Implementation of Algorithm 2 from GEPA paper:
        1. Get candidates that achieve best score on at least one training task (provided by CandidatePool)
        2. Exclude candidates that have already been used for generation (had_child = True)
        3. Prune strictly dominated candidates
        4. Return Pareto frontier candidates

        Args:
            task_scores: Dict with task_id -> candidate (best candidate for each task)
        """
        # Collect all unique candidates from the task winners
        unique_candidates = set(task_scores.values())

        if not unique_candidates:
            return []

        # Exclude candidates that have already been used for generation
        eligible_candidates = [c for c in unique_candidates if not c.had_child]

        if not eligible_candidates:
            return []

        # Apply Pareto selection - remove dominated candidates
        pareto_candidates = self._remove_dominated_candidates(eligible_candidates)

        return pareto_candidates

    def filter_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Filter candidates directly without task score data.

        This applies Pareto selection to a direct list of candidates.

        Args:
            candidates: List of candidate objects to filter
        """
        if not candidates:
            return []

        # Exclude candidates that have already been used for generation
        eligible_candidates = [c for c in candidates if not c.had_child]

        if not eligible_candidates:
            return []

        # Apply Pareto selection - remove dominated candidates
        pareto_candidates = self._remove_dominated_candidates(eligible_candidates)

        return pareto_candidates

    def filter_generation(self, gen_id: int, task_scores: Dict[int, Candidate]) -> List[Candidate]:
        """Filter candidates from a specific generation.

        Args:
            gen_id: Generation ID to filter
            task_scores: Task score data for that generation
        """
        # Filter to only candidates from the specified generation
        gen_candidates = {}
        for task_id, candidate in task_scores.items():
            if candidate.generation_number == gen_id:
                gen_candidates[task_id] = candidate

        # Apply standard filter on generation-specific data
        return self.filter_scores(gen_candidates)

    def filter_generation_history(self, gen_id: int, update_index: int, task_scores: Dict[int, Candidate]) -> List[Candidate]:
        """Filter candidates from a specific historical snapshot.

        For Pareto selection, this behaves the same as filter_generation
        since we don't need the update index for our algorithm.

        Args:
            gen_id: Generation ID
            update_index: Index of the historical snapshot (unused for Pareto)
            task_scores: Historical task score data
        """
        # For Pareto selection, we treat historical data the same way
        return self.filter_generation(gen_id, task_scores)

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
