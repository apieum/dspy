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

import logging
import random
from typing import List, Optional, Dict, TYPE_CHECKING
from collections import defaultdict

import dspy
from .selector import Selector
from ..data.candidate import Candidate
from ..data.cohort import Survivors, Parents
from ..budget import Budget

if TYPE_CHECKING:
    from ..config import DarwinConfig

from ..evaluation import Metric
from .archive import DiversityArchive

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
    - Cohort integration with efficient filtering algorithms
    - Full Selection interface compatibility for system integration
    - Comprehensive logging and error handling
    """

    def __init__(self):
        """Initialize the Pareto Frontier selector."""
        super().__init__()
        # Internal candidate and score management using UUID-based identification
        self.example_best_scores: Dict[str, List["Metric"]] = {}  # example_uuid -> best scores for that example
        self.example_best_candidates: Dict[str, List[Candidate]] = {}  # example_uuid -> candidates with best scores
        self.task_wins: Dict[Candidate, int] = defaultdict(int)  # candidate -> number of examples won
        self.elitist_pruning = False
        self.diversity_archive = None
        self.frontier_type = "instance"
    def configure(self, config: "DarwinConfig") -> None:
        self.frontier_type = getattr(config, "frontier_type", self.frontier_type)

    def start_compilation(self, student: dspy.Module, verbose: bool=False) -> None:
        """Reset all selection state at the beginning of a compilation."""
        self.verbose = verbose
        self.example_best_scores.clear()
        self.example_best_candidates.clear()
        self.task_wins.clear()
        if self.diversity_archive is not None:
            self.diversity_archive = DiversityArchive(self.diversity_archive.capacity)


    def promote(self, survivors: Survivors, budget: Optional[Budget] = None) -> Parents:
        """
        Paper-compliant promotion: Combines new survivors with existing internal population,
        filters for Pareto frontier, and returns the complete parent population.

        Args:
            new_survivors: New candidates that have been fully evaluated
            budget: Optional budget constraints

        Returns:
            Parents cohort containing the Pareto frontier from the union of old and new
        """
        # Report Pareto frontier promotion
        self.publish('promote', survivors)

        # 1. Update scores for new survivors (adds them to internal tracking)
        self.update_scores_batch(survivors)
        if self.diversity_archive is not None:
            self.diversity_archive.add_all(survivors)

        # 2. Get all candidates currently tracked
        pareto_frontier = set()
        for example_winners in self.example_best_candidates.values():
            pareto_frontier.update(example_winners)
        if self.diversity_archive is not None:
            pareto_frontier.update(self.diversity_archive.candidates())

        self.publish('pareto_filtering', {'frontier_size': len(pareto_frontier)})

        # The per-task winner union is a candidate frontier, but it can still
        # contain a candidate globally dominated on another task. Official
        # GEPA removes those candidates before parent sampling.
        pareto_frontier = self._remove_dominated(pareto_frontier)

        # 4. Extract task_wins for the Pareto-filtered candidates
        relevant_task_wins = {
            candidate: self.task_wins[candidate]
            for candidate in pareto_frontier
        }

        result = Parents(
            *pareto_frontier,
            iteration=survivors.iteration + 1,
            task_wins=relevant_task_wins
        )

        # Observer notification already handled above

        return result

    @staticmethod
    def _remove_dominated(candidates):
        candidates = set(candidates)
        return {
            candidate for candidate in candidates
            if not any(
                other is not candidate and other.dominate(candidate)
                for other in candidates
            )
        }

    def best_candidate(self) -> Candidate:
        """Return the best candidate from the pool."""
        if not self.task_wins:
            raise RuntimeError("No candidates found in selector - optimization failed")

        # Get best candidates from task scores and select overall best
        best_candidates = self._remove_dominated(self.task_wins.keys())

        best = max(best_candidates, key=lambda c: c.total_score())

        # Report candidate pool and final selection
        self.publish('best_candidate_selection', best, {'pool_size': len(best_candidates)})

        # Report final optimized instruction
        predictors = best.module.predictors()
        if predictors:
            from dspy.teleprompt.utils import get_signature
            instruction = get_signature(predictors[0]).instructions
            self.publish('final_instruction', {'instruction': instruction})

        return best

    def select_candidate(self, strategy: str = "pareto", rng=None) -> Candidate:
        """Select a final candidate using GEPA's configurable selectors."""
        if not self.task_wins:
            raise RuntimeError("No candidates found in selector - optimization failed")
        rng = rng or random.Random(0)
        candidates = list(self._remove_dominated(self.task_wins))

        if strategy == "current_best":
            return max(candidates, key=lambda c: c.total_score())
        if strategy == "epsilon_greedy":
            if rng.random() < 0.1:
                return rng.choice(candidates)
            return max(candidates, key=lambda c: c.total_score())
        if strategy == "top_k_pareto":
            candidates = sorted(candidates, key=lambda c: c.total_score(), reverse=True)
            candidates = candidates[: min(3, len(candidates))]
        elif strategy != "pareto":
            raise ValueError(f"Unknown candidate selection strategy: {strategy}")

        # GEPA samples Pareto candidates proportional to the number of
        # validation examples each candidate wins, preserving specialists.
        return rng.choices(
            candidates, weights=[self.task_wins[c] for c in candidates], k=1
        )[0]

    def update_score(self, example_uuid: str, candidate: Candidate, score: "Metric") -> None:
        """Update the example scores with a candidate for a specific example UUID."""
        current_winners = self.example_best_candidates.get(example_uuid, [])
        current_best_value = self.example_best_scores.get(example_uuid, [Metric(0.0)])[0]

        if not current_winners:
            self.example_best_candidates[example_uuid] = [candidate]
            self.example_best_scores[example_uuid] = [score]
            self.task_wins[candidate] += 1
        elif score > current_best_value:
            # New candidate is strictly better on this example → replace all current winners.
            self._update_old_winners(current_winners)
            self.example_best_candidates[example_uuid] = [candidate]
            self.example_best_scores[example_uuid] = [score]
            self.task_wins[candidate] += 1
        elif score == current_best_value:
            # Tied performance. Decide how to handle the tie.
            if self.elitist_pruning:
                # Elitist mode: Use dominance and ancestry to prune the winner set.
                self._handle_tie_with_elitism(example_uuid, candidate, score, current_winners)
            else:
                # Official GEPA mode: Add the new candidate to the set of winners.
                self.example_best_candidates[example_uuid].append(candidate)
                self.example_best_scores[example_uuid].append(score)
                self.task_wins[candidate] += 1
        # else: score < current_best_value → ignore candidate

        self.publish('update_score', candidate, score)

    def _handle_tie_with_elitism(self, example_uuid: str, candidate: Candidate, score: "Metric", current_winners: List[Candidate]):
        """Handle a tie in scores using dominance and ancestry checks."""
        dominates_any = any(candidate.dominate(winner) for winner in current_winners)
        dominated_by_any = any(winner.dominate(candidate) for winner in current_winners)

        if dominates_any and not dominated_by_any:
            # New candidate dominates at least one winner and is not dominated itself.
            dominated_winners = [w for w in current_winners if candidate.dominate(w)]
            non_dominated = [w for w in current_winners if not candidate.dominate(w)]
            self._update_old_winners(dominated_winners)
            self.example_best_candidates[example_uuid] = non_dominated + [candidate]
            self.example_best_scores[example_uuid].append(score)
            self.task_wins[candidate] += 1
        elif not dominated_by_any:
            # Not dominated by any existing winner. Now check ancestry.
            parents_in_winners = [w for w in current_winners if w.is_ancestor_of(candidate)]
            if parents_in_winners:
                # Child replaces its parents if it performs equally well.
                self._update_old_winners(parents_in_winners)
                self.example_best_candidates[example_uuid] = [w for w in current_winners if w not in parents_in_winners] + [candidate]
                self.example_best_scores[example_uuid].append(score)
                self.task_wins[candidate] += 1
            elif not candidate.is_ancestor_of_any(current_winners):
                # No ancestry relationship and not dominated, so it's a genuinely different solution.
                self.example_best_candidates[example_uuid].append(candidate)
                self.example_best_scores[example_uuid].append(score)
                self.task_wins[candidate] += 1


    def _update_old_winners(self, old_winners):
        for old_winner in old_winners:
            if self.task_wins[old_winner] <= 1:
                del self.task_wins[old_winner]
            else:
                self.task_wins[old_winner] -= 1

    def update_scores_batch(self, candidates: Survivors) -> None:
        """Update fitness scores for multiple candidates efficiently.

        This method processes all candidates and all their scores in one go,
        avoiding redundant individual update_score calls.

        Args:
            candidates: List of candidates to process
        """
        if not candidates:
            return

        for candidate in candidates:
            for score in candidate.scores:
                for frontier_key, value in self._frontier_entries(score):
                    frontier_score = score if frontier_key == score.id else Metric(value, id=frontier_key)
                    self.update_score(frontier_key, candidate, frontier_score)


        self.publish('update_scores_batch', candidates)

    def configure(self, config: 'DarwinConfig') -> None:
        """Configure the selector with observers and settings.

        Args:
            config: Configuration containing observers and settings
        """
        self.elitist_pruning = getattr(config, 'elitist_pruning', False)
        self.frontier_type = getattr(config, 'frontier_type', 'instance')
        if getattr(config, 'preserve_diversity', False):
            self.diversity_archive = DiversityArchive(getattr(config, 'archive_capacity', 32))
        else:
            self.diversity_archive = None
        # Subscribe all selector observers to our events using the Channel pattern
        for observer in getattr(config, 'selector_observers', []):
            # Use the modern Channel subscription pattern
            for event_name in ['promote', 'update_score', 'update_scores_batch']:
                if hasattr(observer, event_name):
                    self.subscribe(observer, event_name)

    def _frontier_entries(self, score: "Metric"):
        """Expand scalar/objective metrics into configured frontier keys."""
        objectives = getattr(score, "objective_scores", {}) or {}
        if self.frontier_type == "instance" or not objectives:
            return [(score.id, float(score.value))]
        objective_entries = [(f"objective:{name}", value) for name, value in objectives.items()]
        if self.frontier_type == "objective":
            return objective_entries
        if self.frontier_type == "cartesian":
            return [(f"{key}:{score.id}", value) for key, value in objective_entries]
        return [(score.id, float(score.value)), *objective_entries]
