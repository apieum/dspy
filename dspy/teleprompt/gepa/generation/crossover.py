"""System-Aware Merge generator implementing Algorithm 4 from the GEPA paper.

This is a self-contained implementation that integrates all necessary logic
for ancestry, desirability checks, and history tracking, simplifying the
overall architecture and removing the need for a separate utils directory.
"""

import logging
from typing import List, Optional, Tuple, Set, TYPE_CHECKING

import dspy
from dspy.teleprompt.utils import get_signature, set_signature
from .generator import Generator
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns

if TYPE_CHECKING:
    from ..dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class SystemAwareMerge(Generator):
    """
    System-Aware Merge generator implementing Algorithm 4 from the GEPA paper.

    This is a self-contained implementation that integrates all necessary logic
    for ancestry, desirability checks, and history tracking, simplifying the
    overall architecture and removing the need for a separate utils directory.
    """

    def __init__(self, merge_rate: float = 0.7, population_size: int = 10):
        self.merge_rate = merge_rate
        self.population_size = population_size

        # DatasetManager for interface compatibility (not used in Algorithm 4)
        self.dataset_manager: Optional["DatasetManager"] = None

        # Integrated merge history tracking (replaces MergeHistoryTracker)
        self.attempted_merges: Set[Tuple[int, int, int]] = set()
        self.merge_stats = {"success": 0, "failure_not_desirable": 0, "failure_ancestry": 0}

    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate a new candidate using System-Aware Merge (Algorithm 4)."""
        if parents.size() < 2:
            return NewBorns()

        try:
            # Step 1: Stochastic selection of two parent candidates
            selected_parents = parents.sample_stochastic(2)
            if selected_parents.size() < 2:
                return NewBorns()

            parent1, parent2 = list(selected_parents)

            # Step 2: Check for direct ancestry
            # This is much cleaner and respects encapsulation.
            if parent1.is_ancestor_of(parent2) or parent2.is_ancestor_of(parent1):
                self.merge_stats["failure_ancestry"] += 1
                return NewBorns()

            # Step 3: Find common ancestors
            common_ancestors = parent1.find_common_ancestors(parent2)
            if not common_ancestors:
                return NewBorns()

            # Step 4: Iterate through common ancestors to find a valid merge
            # Sort by generation number (most recent first) for better results
            for ancestor in sorted(list(common_ancestors), key=lambda c: c.generation_number, reverse=True):

                # Check merge history (integrated logic)
                merge_key = tuple(sorted((id(parent1), id(parent2)))) + (id(ancestor),)
                if merge_key in self.attempted_merges:
                    continue
                self.attempted_merges.add(merge_key)

                # Step 5: Check for desirable divergence (integrated as a private method)
                desirable_signatures = self._find_desirable_signatures(ancestor, parent1, parent2)
                if not desirable_signatures:
                    self.merge_stats["failure_not_desirable"] += 1
                    continue

                # Step 6: Create the merged candidate
                child_candidate = self._create_merged_candidate(
                    ancestor, parent1, parent2, parents.iteration, desirable_signatures
                )

                self.merge_stats["success"] += 1
                return NewBorns(child_candidate, iteration=parents.iteration)

            logger.debug(f"No successful merge found for parents despite {len(common_ancestors)} common ancestors")

        except Exception as e:
            logger.warning(f"System-Aware Merge failed: {e}")

        return NewBorns()  # No successful merge found

    def _find_desirable_signatures(self, ancestor: Candidate, p1: Candidate, p2: Candidate) -> List[Tuple[int, any]]:
        """
        Private method to check for desirable signature patterns.
        This encapsulates the logic from the 'desirable' function.

        Implements the DESIRABLE function from Algorithm 4:
        - Condition 1: p1 innovated, p2 did not (πa = πj and πi ≠ πj) → use p1's innovation
        - Condition 2: p2 innovated, p1 did not (πa = πi and πj ≠ πi) → use p2's innovation
        - Condition 3: Both innovated differently → pick from better-performing parent
        """
        selected_signatures = []

        try:
            ancestor_predictors = ancestor.module.predictors()
            p1_predictors = p1.module.predictors()
            p2_predictors = p2.module.predictors()

            if not ancestor_predictors:
                return []

            # Iterate through modules, assuming lists are of the same length
            max_modules = max(len(ancestor_predictors), len(p1_predictors), len(p2_predictors))

            for i in range(max_modules):
                # Get signatures (skip if any is missing)
                if (i >= len(ancestor_predictors) or
                    i >= len(p1_predictors) or
                    i >= len(p2_predictors)):
                    continue

                pred_a, pred_p1, pred_p2 = ancestor_predictors[i], p1_predictors[i], p2_predictors[i]
                sig_a, sig_p1, sig_p2 = get_signature(pred_a), get_signature(pred_p1), get_signature(pred_p2)

                # Get signature strings for comparison
                π_a = sig_a.signature if hasattr(sig_a, 'signature') else str(sig_a)
                π_p1 = sig_p1.signature if hasattr(sig_p1, 'signature') else str(sig_p1)
                π_p2 = sig_p2.signature if hasattr(sig_p2, 'signature') else str(sig_p2)

                # Condition 1: p1 innovated, p2 did not (πa = πp2 and πp1 ≠ πp2)
                if π_a == π_p2 and π_a != π_p1:
                    selected_signatures.append((i, sig_p1))
                    logger.debug(f"DESIRABLE: Module {i} - p1 innovated, using p1's signature")

                # Condition 2: p2 innovated, p1 did not (πa = πp1 and πp2 ≠ πp1)
                elif π_a == π_p1 and π_a != π_p2:
                    selected_signatures.append((i, sig_p2))
                    logger.debug(f"DESIRABLE: Module {i} - p2 innovated, using p2's signature")

                # Condition 3: Both innovated differently, pick from the better-performing parent
                elif π_a != π_p1 and π_a != π_p2 and π_p1 != π_p2:
                    best_parent = p1 if p1.average_task_score() > p2.average_task_score() else p2
                    selected_signature = sig_p1 if best_parent == p1 else sig_p2
                    selected_signatures.append((i, selected_signature))
                    logger.debug(f"DESIRABLE: Module {i} - both innovated, using {'p1' if best_parent == p1 else 'p2'}'s signature")

        except Exception as e:
            logger.warning(f"Error finding desirable signatures: {e}")
            return []

        return selected_signatures

    def _create_merged_candidate(self, ancestor: Candidate, p1: Candidate, p2: Candidate,
                               iteration: int, signatures_to_apply: List[Tuple[int, any]]) -> Candidate:
        """Creates a new candidate by merging parent innovations onto an ancestor."""
        try:
            child_module = ancestor.module.deepcopy()
            child_predictors = child_module.predictors()

            for module_idx, new_signature in signatures_to_apply:
                if module_idx < len(child_predictors):
                    set_signature(child_predictors[module_idx], new_signature)
                    logger.debug(f"Applied selected signature to module {module_idx}")

            return Candidate(
                module=child_module,
                parents=[p1, p2, ancestor],  # 3-way lineage
                generation_number=iteration,
                creation_metadata={
                    "merge_type": "system_aware",
                    "ancestor_generation": ancestor.generation_number,
                    "parent1_generation": p1.generation_number,
                    "parent2_generation": p2.generation_number,
                    "modules_merged": len(signatures_to_apply)
                }
            )

        except Exception as e:
            logger.warning(f"Error creating merged candidate: {e}")
            return None

    def start_compilation(self, student: dspy.Module, dataset_manager: "DatasetManager") -> None:
        """Resets merge history for a new compilation run."""
        # Store dataset manager for interface compatibility
        # (Algorithm 4 doesn't use it, unlike ReflectivePromptMutation)
        self.dataset_manager = dataset_manager

        self.attempted_merges.clear()
        self.merge_stats = {"success": 0, "failure_not_desirable": 0, "failure_ancestry": 0}
        logger.debug("Reset SystemAwareMerge for new compilation")

    def get_merge_statistics(self) -> dict:
        """Get statistics about merge attempts for debugging/monitoring."""
        total_attempts = sum(self.merge_stats.values())
        success_rate = self.merge_stats["success"] / total_attempts if total_attempts > 0 else 0.0

        return {
            "total_attempts": total_attempts,
            "successful_merges": self.merge_stats["success"],
            "failed_not_desirable": self.merge_stats["failure_not_desirable"],
            "failed_ancestry": self.merge_stats["failure_ancestry"],
            "success_rate": success_rate,
            "unique_combinations_attempted": len(self.attempted_merges)
        }

    def clear_merge_history(self) -> None:
        """Clear merge history (useful for testing or reset)."""
        self.attempted_merges.clear()
        self.merge_stats = {"success": 0, "failure_not_desirable": 0, "failure_ancestry": 0}
        logger.debug("Cleared SystemAwareMerge history")
