"""GEPA Algorithm 2: Pareto Frontier Selection - The Official Implementation.

This is the primary, paper-compliant implementation of Pareto frontier selection
from the GEPA paper. It efficiently computes the Pareto frontier using advanced
accumulator patterns and stochastic sampling based on task winning frequency.

Key Features:
- Exact implementation of GEPA Algorithm 2
- Efficient accumulator pattern for task winners
- Advanced stochastic sampling with winning frequency weighting
- Modern Cohort integration with efficient filtering
- Comprehensive Selection interface compatibility
"""

import random
import logging
from typing import List, Set, Optional, Dict, Callable, TYPE_CHECKING
from collections import defaultdict

from .selector import Selector
from ..data.candidate import Candidate
from ..data.cohort import Cohort, Survivors, Parents
from ..budget import Budget

if TYPE_CHECKING:
    from ..dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class ParetoFrontier(Selector):
    """The Official GEPA Algorithm 2: Pareto Frontier Selection.

    This is the primary implementation from the GEPA paper, featuring:

    **Algorithm 2 Implementation:**
    1. Accumulate candidates that achieve best score on at least one task
    2. Remove strictly dominated candidates using efficient Pareto filtering
    3. Support stochastic sampling based on task winning frequency

    **Advanced Features:**
    - Accumulator pattern for optimal task winner collection
    - Frequency-weighted stochastic sampling for diversity
    - Modern Cohort integration with efficient filtering algorithms
    - Full Selection interface compatibility for system integration
    - Comprehensive logging and error handling

    **Performance Optimizations:**
    - Uses Cohort.filter() with pop-based domination removal
    - Efficient task winner accumulation with O(1) lookups
    - Lazy evaluation of Pareto frontier computation
    """

    def __init__(self):
        """Initialize the Pareto Frontier selector."""
        self.selection_stats = defaultdict(int)  # Statistics tracking
        # Internal candidate and task score management
        self.task_scores: Dict[int, float] = {}  # task_id -> score for that task
        self.task_best_candidates: Dict[int, List[Candidate]] = {}  # task_id -> all tied winners
        self.task_wins: Dict[Candidate, int] = defaultdict(int)  # candidate -> number of tasks won
        self.dataset_manager: Optional["DatasetManager"] = None

    def start_compilation(self, student, dataset_manager: "DatasetManager") -> None:
        """Called when compilation begins. Initialize task tracking structures."""
        self.dataset_manager = dataset_manager
        num_tasks = self.dataset_manager.num_pareto_tasks
        
        # Initialize task_best_candidates and task_scores for all tasks
        for task_id in range(num_tasks):
            self.task_best_candidates[task_id] = []
            self.task_scores[task_id] = 0.0

    def filter(self, survivors: Survivors, budget: Optional[Budget] = None) -> Survivors:
        """GEPA Algorithm 2: Pareto frontier selection.

        Args:
            survivors: Current cohort of survivors to filter from
            budget: Optional budget constraints

        Returns:
            Cohort containing the Pareto frontier candidates
        """
        self.budget = budget
        # Get all task winner candidates (including ties)
        all_task_winners = set()
        for task_winners in self.task_best_candidates.values():
            all_task_winners.update(task_winners)

        # Filter out candidates that are not eligible parents (from previous iteration)
        eligible_candidates = [c for c in all_task_winners if c.generation_number == survivors.iteration - 1]

        # Remove Pareto-dominated candidates
        pareto_frontier = self._remove_dominated_candidates(eligible_candidates)

        return Survivors(*pareto_frontier)

    def promote(self, new_survivors: Survivors, budget: Optional[Budget] = None) -> Parents:
        """
        Paper-compliant promotion: Combines new survivors with existing internal population,
        filters for Pareto frontier, and returns the complete parent population.
        
        Args:
            new_survivors: New candidates that have been fully evaluated
            budget: Optional budget constraints

        Returns:
            Parents cohort containing the Pareto frontier from the union of old and new
        """
        # 1. Update scores for new survivors (adds them to internal tracking)
        self.update_scores_batch(new_survivors)
        
        # 2. Get all candidates currently tracked (includes both old and new)
        all_tracked_candidates = set()
        for task_winners in self.task_best_candidates.values():
            all_tracked_candidates.update(task_winners)
        
        logger.debug(f"Promoting: {len(all_tracked_candidates)} total tracked candidates for Pareto filtering")

        # 3. Filter the combined pool to find the true Pareto frontier
        pareto_frontier = self._remove_dominated_candidates(list(all_tracked_candidates))
        logger.debug(f"Pareto frontier contains {len(pareto_frontier)} candidates after filtering")

        # 4. Extract task_wins for the Pareto-filtered candidates
        relevant_task_wins = {
            candidate: self.task_wins[candidate]
            for candidate in pareto_frontier
        }

        return Parents(
            *pareto_frontier,
            iteration=new_survivors.iteration + 1,
            task_wins=relevant_task_wins
        )

    def best_candidate(self) -> Candidate:
        """Return the best candidate from the pool."""
        if not self.task_wins:
            raise RuntimeError("No candidates found in selector - optimization failed")

        # Get best candidates from task scores and select overall best
        best_candidates = self.task_wins.keys()
        return max(best_candidates, key=lambda c: c.average_task_score())

    def update_score(self, task_id: int, candidate: Candidate) -> None:
        """Update the task scores with a candidate for a specific task."""
        current_winners = self.task_best_candidates.get(task_id, [])

        if not current_winners:
            # First candidate for this task
            self.task_best_candidates[task_id] = [candidate]
            self.task_scores[task_id] = candidate.task_score(task_id)
            self.task_wins[candidate] = self.task_wins.get(candidate, 0) + 1
            return

        # For task winners, we first check performance on THIS SPECIFIC TASK
        candidate_score = candidate.task_score(task_id)
        current_best_score = self.task_scores[task_id]

        if candidate_score > current_best_score:
            # New candidate is strictly better on this task → replace all current winners
            self.task_best_candidates[task_id] = [candidate]
            self.task_scores[task_id] = candidate.task_score(task_id)
            self.task_wins[candidate] = self.task_wins.get(candidate, 0) + 1
            for old_winner in current_winners:
                self.task_wins[old_winner] -= 1
        elif candidate_score == current_best_score:
            # Tied performance → check domination across ALL tasks and ancestry
            dominates_any = any(candidate.better_than(winner) for winner in current_winners)
            dominated_by_any = any(winner.better_than(candidate) for winner in current_winners)

            if dominates_any:
                # New candidate dominates at least one winner → replace all dominated ones
                dominated_winners = [w for w in current_winners if candidate.better_than(w)]
                non_dominated = [w for w in current_winners if not candidate.better_than(w)]

                # Update task_wins: subtract from dominated, add to new candidate
                for dominated in dominated_winners:
                    self.task_wins[dominated] -= 1
                self.task_wins[candidate] = self.task_wins.get(candidate, 0) + 1

                self.task_best_candidates[task_id] = non_dominated + [candidate]
            elif dominated_by_any:
                # New candidate is dominated → ignore it
                return
            else:
                # Neither dominates → check ancestry
                parents_in_winners = [w for w in current_winners if self._is_ancestor(w, candidate)]

                if parents_in_winners:
                    # Remove all parents and add the evolved child
                    for parent in parents_in_winners:
                        current_winners.remove(parent)
                        self.task_wins[parent] -= 1
                    current_winners.append(candidate)
                    self.task_wins[candidate] = self.task_wins.get(candidate, 0) + 1
                elif not self._has_any_descendants_in(candidate, current_winners):
                    # No ancestry relationship → keep as genuinely different solution
                    current_winners.append(candidate)
                    self.task_wins[candidate] = self.task_wins.get(candidate, 0) + 1
        # else: candidate_score < current_best_score → ignore candidate

    def update_scores_batch(self, candidates: Survivors) -> None:
        """Update task scores for multiple candidates efficiently.

        This method processes all candidates and all their tasks in one go,
        avoiding redundant individual update_score calls.

        Args:
            candidates: List of candidates to process
        """
        if not candidates:
            return

        # Get all task IDs from all candidates
        all_task_ids = set(self.task_scores.keys())

        # Process each task once
        for task_id in all_task_ids:
            for candidate in candidates:
                self.update_score(task_id, candidate)

    def _is_ancestor(self, potential_ancestor: Candidate, descendant: Candidate) -> bool:
        """Check if potential_ancestor is an ancestor of descendant."""
        found_ancestor = False
        def check_ancestor(ancestor, gen_number, metadata):
            nonlocal found_ancestor
            if ancestor == potential_ancestor:
                found_ancestor = True
                return False  # Stop traversal
            return True  # Continue traversal

        descendant.traverse_ancestors(check_ancestor)
        return found_ancestor

    def _has_any_descendants_in(self, candidate: Candidate, candidate_list: List[Candidate]) -> bool:
        """Check if candidate has any descendants in the given list."""
        for other_candidate in candidate_list:
            if self._is_ancestor(candidate, other_candidate):
                return True
        return False

    def _pareto_filter(self, candidates: List[Candidate], survivors: Cohort) -> List[Candidate]:
        """Filter candidates using Pareto dominance.

        Candidates passed here are already the best for at least one task.
        We just need to filter out used candidates and remove dominated ones.
        """
        # Filter out candidates that are not eligible parents
        eligible_candidates = [c for c in candidates if c.generation_number == survivors.iteration - 1]

        if not eligible_candidates:
            return []

        # Remove dominated candidates
        return self._remove_dominated_candidates(eligible_candidates)




    def _remove_dominated_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Remove strictly dominated candidates using Pareto dominance."""
        if not candidates:
            return []

        pareto_frontier = []

        for candidate in candidates:
            is_dominated = False

            # Check if this candidate is dominated by any candidate already in the frontier
            for frontier_candidate in pareto_frontier:
                if frontier_candidate.better_than(candidate):
                    is_dominated = True
                    break

            if not is_dominated:
                # Remove any frontier candidates that are dominated by this candidate
                pareto_frontier = [fc for fc in pareto_frontier if not candidate.better_than(fc)]
                pareto_frontier.append(candidate)

        return pareto_frontier


    def get_selection_stats(self) -> dict:
        """Get selection statistics for monitoring and debugging."""
        return dict(self.selection_stats)
