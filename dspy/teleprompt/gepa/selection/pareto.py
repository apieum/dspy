"""Pareto-based candidate selection implementing Algorithm 2."""

import logging
import random
from collections import defaultdict
from typing import List

import dspy

from ..data.structures import ScoreMatrix
from .base import CandidateSelector

logger = logging.getLogger(__name__)


class ParetoCandidateSelector(CandidateSelector):
    """Pareto-based candidate selection (Algorithm 2)."""

    def __init__(self):
        self.selection_counts = defaultdict(int)

    def select_candidate(self, candidates: List[dspy.Module], scores: ScoreMatrix) -> int:
        """Select candidate using Pareto-based illumination strategy (Algorithm 2).

        Implementation of Algorithm 2 from GEPA paper:
        1. Identify highest score for each training instance across all candidates
        2. Compile candidates that achieve best score on at least one training task
        3. Prune strictly dominated candidates
        4. Stochastically sample from remaining candidates based on their "winning" frequency
        """
        if not candidates:
            return 0

        # Get candidates with scores
        candidate_indices = list(range(len(candidates)))
        scored_candidates = [idx for idx in candidate_indices if scores.get_candidate_scores(idx)]

        if not scored_candidates:
            # No candidates have scores yet, select first
            return 0

        if len(scored_candidates) == 1:
            # Only one candidate scored, select it
            self.selection_counts[scored_candidates[0]] += 1
            return scored_candidates[0]

        # Algorithm 2: Pareto-based selection
        pareto_candidates = self._find_pareto_frontier(scored_candidates, scores)
        selected_idx = self._stochastic_sample_from_pareto(pareto_candidates, scores)

        self.selection_counts[selected_idx] += 1
        return selected_idx

    def _find_pareto_frontier(self, candidate_indices: List[int], scores: ScoreMatrix) -> List[int]:
        """Find Pareto frontier of candidates based on task-level performance.

        A candidate is in the Pareto frontier if it achieves the best score
        on at least one training task, and is not strictly dominated.
        """
        # Get all task indices by examining score matrix
        all_task_indices = set()
        for candidate_idx in candidate_indices:
            all_task_indices.update(scores.get_candidate_scores(candidate_idx).keys())
        all_task_indices = list(all_task_indices)

        if not all_task_indices:
            return candidate_indices[:1]  # Fallback to first candidate

        # Step 1: Find best score for each task
        task_best_scores = {}
        for task_idx in all_task_indices:
            best_score = -float('inf')
            for candidate_idx in candidate_indices:
                score = scores.get_score(candidate_idx, task_idx)
                if score is not None and score > best_score:
                    best_score = score
            task_best_scores[task_idx] = best_score

        # Step 2: Find candidates that achieve best score on at least one task
        winning_candidates = set()
        candidate_wins = defaultdict(list)  # candidate_idx -> list of tasks where it wins

        for task_idx in all_task_indices:
            best_score = task_best_scores[task_idx]
            for candidate_idx in candidate_indices:
                score = scores.get_score(candidate_idx, task_idx)
                if score is not None and abs(score - best_score) < 1e-6:  # Account for float precision
                    winning_candidates.add(candidate_idx)
                    candidate_wins[candidate_idx].append(task_idx)

        if not winning_candidates:
            return candidate_indices[:1]  # Fallback

        # Step 3: Prune strictly dominated candidates
        pareto_candidates = list(winning_candidates)
        pareto_candidates = self._remove_dominated_candidates(pareto_candidates, scores, all_task_indices)

        return pareto_candidates if pareto_candidates else candidate_indices[:1]

    def _remove_dominated_candidates(self, candidates: List[int], scores: ScoreMatrix, task_indices: List[int]) -> List[int]:
        """Remove strictly dominated candidates.

        Candidate A dominates candidate B if A performs >= B on all tasks
        and A performs > B on at least one task.
        """
        non_dominated = []

        for i, candidate_a in enumerate(candidates):
            is_dominated = False

            for j, candidate_b in enumerate(candidates):
                if i == j:
                    continue

                # Check if candidate_b dominates candidate_a
                dominates = True
                strictly_better_on_some = False

                for task_idx in task_indices:
                    score_a = scores.get_score(candidate_a, task_idx) or 0.0
                    score_b = scores.get_score(candidate_b, task_idx) or 0.0

                    if score_b < score_a:
                        dominates = False
                        break
                    elif score_b > score_a:
                        strictly_better_on_some = True

                if dominates and strictly_better_on_some:
                    is_dominated = True
                    break

            if not is_dominated:
                non_dominated.append(candidate_a)

        return non_dominated

    def _stochastic_sample_from_pareto(self, pareto_candidates: List[int], scores: ScoreMatrix) -> int:
        """Stochastically sample from Pareto frontier based on winning frequency.

        Candidates with higher winning frequency (more tasks where they achieve best score)
        are more likely to be selected.
        """
        if len(pareto_candidates) == 1:
            return pareto_candidates[0]

        # Count wins for each candidate
        candidate_wins = defaultdict(int)
        all_task_indices = set()

        for candidate_idx in pareto_candidates:
            all_task_indices.update(scores.get_candidate_scores(candidate_idx).keys())
        all_task_indices = list(all_task_indices)

        # Count wins per candidate
        for task_idx in all_task_indices:
            best_score = -float('inf')
            best_candidates = []

            for candidate_idx in pareto_candidates:
                score = scores.get_score(candidate_idx, task_idx) or 0.0
                if score > best_score:
                    best_score = score
                    best_candidates = [candidate_idx]
                elif abs(score - best_score) < 1e-6:
                    best_candidates.append(candidate_idx)

            # Award wins to tied candidates
            for candidate_idx in best_candidates:
                candidate_wins[candidate_idx] += 1.0 / len(best_candidates)

        # Convert to probabilities
        total_wins = sum(candidate_wins.values())
        if total_wins == 0:
            # Uniform selection if no wins computed
            return random.choice(pareto_candidates)

        probabilities = [candidate_wins[candidate_idx] / total_wins for candidate_idx in pareto_candidates]

        # Stochastic selection based on winning frequency
        cumulative_prob = 0.0
        rand_val = random.random()

        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return pareto_candidates[i]

        # Fallback (should not reach here)
        return pareto_candidates[-1]