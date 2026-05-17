"""System-Aware Merge - Context-aware evolutionary crossover from the GEPA paper.

This implements Algorithm 4 from the GEPA paper with integrated logic for
ancestry tracking, desirability analysis, and signature merging.
"""

import logging
from typing import List, Optional, Tuple, Set, TYPE_CHECKING

import dspy
from dspy.teleprompt.utils import get_signature, set_signature
from .generator import Generator
from .mutation import ReflectivePromptMutation
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns


logger = logging.getLogger(__name__)


class SystemAwareMerge(Generator):
    """
    System-Aware Merge generator implementing Algorithm 4 from the GEPA paper.
    """

    def __init__(self, feedback_provider=None, feedback_data=None, assessor=None, config=None):
        # Integrated merge history tracking (replaces MergeHistoryTracker)
        self.attempted_merges: Set[Tuple[int, int, int]] = set()
        self.merge_stats = {"success": 0, "failure_not_desirable": 0, "failure_ancestry": 0}
        self.verbose = False
        self.feedback_provider = feedback_provider
        self.assessor = assessor or getattr(feedback_provider, "assessor", None)
        self.config = config

        # Initialize fallback mutation generator
        self.fallback_generator = None
        self.devset = feedback_data or []

    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate a new candidate using System-Aware Merge (Algorithm 4)."""
        if parents.size() < 2:
            # Fall back to ReflectivePromptMutation when there aren't enough parents for merge
            logger.debug(f"SystemAwareMerge: Only {parents.size()} parents available, falling back to ReflectivePromptMutation")
            if self.fallback_generator and self.fallback_generator.feedback_data:
                return self.fallback_generator.generate(parents, budget)
            else:
                logger.warning("SystemAwareMerge: Fallback generator not initialized")
                return NewBorns()

        try:
            # Stochastic selection of two parent candidates
            selected_parents = parents.sample_stochastic(2)
            if selected_parents.size() < 2:
                return NewBorns()

            parent1, parent2 = list(selected_parents)

            # Find common ancestors
            common_ancestors = parent1.find_common_ancestors(parent2)

            # Iterate through common ancestors to find a valid merge
            # Sort by generation number (most recent first) for better results
            for ancestor in sorted(list(common_ancestors), key=lambda c: c.generation_number, reverse=True):

                # Check merge history (integrated logic)
                merge_key = tuple(sorted((id(parent1), id(parent2)))) + (id(ancestor),)
                if merge_key in self.attempted_merges:
                    continue
                self.attempted_merges.add(merge_key)

                # Check for desirable divergence
                desirable_signatures = self._find_desirable_signatures(ancestor, parent1, parent2)
                if not desirable_signatures:
                    self.merge_stats["failure_not_desirable"] += 1
                    continue

                # Step 6: Create the merged candidate
                child_candidate = self._create_merged_candidate(
                    ancestor, parent1, parent2, parents.iteration, desirable_signatures
                )

                # Display merge evolution if verbose mode is enabled
                if self.verbose:
                    self._display_merge_evolution(ancestor, parent1, parent2, child_candidate, desirable_signatures)

                self.merge_stats["success"] += 1
                return NewBorns(child_candidate, iteration=parents.iteration)

            logger.debug(f"No successful merge found for parents despite {len(common_ancestors)} common ancestors")

        except Exception as e:
            logger.warning(f"System-Aware Merge failed: {e}")

        return NewBorns()  # No successful merge found

    def _find_desirable_signatures(self, ancestor: Candidate, p1: Candidate, p2: Candidate) -> List[Tuple[int, any]]:
        """
        Check for desirable signature patterns.
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
                    best_parent = p1 if p1.average_score() > p2.average_score() else p2
                    selected_signature = sig_p1 if best_parent == p1 else sig_p2
                    selected_signatures.append((i, selected_signature))
                    logger.debug(f"DESIRABLE: Module {i} - both innovated, using {'p1' if best_parent == p1 else 'p2'}'s signature")

        except Exception as e:
            logger.warning(f"Error finding desirable signatures: {e}")
            return []

        return selected_signatures

    def _display_merge_evolution(self, ancestor: Candidate, parent1: Candidate, parent2: Candidate,
                               child: Candidate, merged_signatures: List[Tuple[int, any]]) -> None:
        """Display system-aware merge evolution in verbose mode with professional formatting."""

        print(f"\n{'='*80}")
        print("SYSTEM-AWARE MERGE EVOLUTION")
        print(f"{'='*80}")
        print(f"Ancestor Generation: {ancestor.generation_number}")
        print(f"Parent 1 Generation: {parent1.generation_number} (Score: {parent1.average_score():.3f})")
        print(f"Parent 2 Generation: {parent2.generation_number} (Score: {parent2.average_score():.3f})")
        print(f"Child Generation: {child.generation_number}")
        print(f"Modules Merged: {len(merged_signatures)}")
        print()

        # Display signature evolution for each merged module
        try:
            ancestor_predictors = ancestor.module.predictors()
            parent1_predictors = parent1.module.predictors()
            parent2_predictors = parent2.module.predictors()
            child_predictors = child.module.predictors()

            for module_idx, new_signature in merged_signatures:
                if (module_idx < len(ancestor_predictors) and
                    module_idx < len(parent1_predictors) and
                    module_idx < len(parent2_predictors) and
                    module_idx < len(child_predictors)):

                    # Get instruction text from signatures
                    ancestor_instr = get_signature(ancestor_predictors[module_idx]).instructions or "No instruction"
                    parent1_instr = get_signature(parent1_predictors[module_idx]).instructions or "No instruction"
                    parent2_instr = get_signature(parent2_predictors[module_idx]).instructions or "No instruction"
                    child_instr = get_signature(child_predictors[module_idx]).instructions or "No instruction"

                    print(f"MODULE {module_idx} SIGNATURE EVOLUTION:")
                    print("-" * 50)
                    print(f"Ancestor:  {ancestor_instr}")
                    print(f"Parent 1:  {parent1_instr}")
                    print(f"Parent 2:  {parent2_instr}")
                    print(f"Child:     {child_instr}")

                    # Determine merge decision logic
                    if ancestor_instr == parent2_instr and ancestor_instr != parent1_instr:
                        print("Decision:  Parent 1 innovated, selected Parent 1 signature")
                    elif ancestor_instr == parent1_instr and ancestor_instr != parent2_instr:
                        print("Decision:  Parent 2 innovated, selected Parent 2 signature")
                    elif ancestor_instr != parent1_instr and ancestor_instr != parent2_instr and parent1_instr != parent2_instr:
                        better_parent = "Parent 1" if parent1.average_score() > parent2.average_score() else "Parent 2"
                        print(f"Decision:  Both parents innovated, selected {better_parent} signature")
                    else:
                        print("Decision:  Standard merge logic applied")

                    print()

        except Exception as e:
            print(f"Error displaying signature details: {e}")

        print("=" * 80)
        print()

    def _create_merged_candidate(self, ancestor: Candidate, p1: Candidate, p2: Candidate,
                               iteration: int, signatures_to_apply: List[Tuple[int, any]]) -> Candidate:
        """Creates a new candidate by merging parent innovations onto an ancestor."""
        try:
            child_module = ancestor.module.deepcopy()
            child_predictors = child_module.predictors()

            for module_idx, new_signature in signatures_to_apply:
                if module_idx < len(child_predictors):
                    # Update both instructions and fields
                    new_instruction = new_signature.instructions if hasattr(new_signature, 'instructions') else str(new_signature)
                    current_signature = get_signature(child_predictors[module_idx])

                    *_, last_key = current_signature.fields.keys()
                    current_prefix = current_signature.fields[last_key].json_schema_extra.get("prefix", "")

                    updated_signature = (
                        current_signature
                        .with_instructions(new_instruction)
                        .with_updated_fields(last_key, prefix=current_prefix)
                    )
                    set_signature(child_predictors[module_idx], updated_signature)
                    logger.debug(f"Applied selected signature to module {module_idx}: {new_instruction}")

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

    def start_compilation(
        self,
        student: dspy.Module,
        *,
        feedback_data: Optional[List[dspy.Example]] = None,
        verbose: bool = False,
    ) -> None:
        """Resets merge history for a new compilation run."""
        self.attempted_merges.clear()
        self.merge_stats = {"success": 0, "failure_not_desirable": 0, "failure_ancestry": 0}
        self.verbose = verbose
        self.devset = feedback_data or []

        # Initialize fallback ReflectivePromptMutation generator
        if self.feedback_provider or self.assessor:
            if self.feedback_provider is None:
                from .feedback import FeedbackProvider
                self.feedback_provider = FeedbackProvider(assessor=self.assessor)
            self.fallback_generator = ReflectivePromptMutation(
                feedback_provider=self.feedback_provider,
                feedback_data=self.devset,
                config=self.config,
            )
            self.fallback_generator.start_compilation(student, feedback_data=self.devset, verbose=verbose)
            logger.debug("Initialized ReflectivePromptMutation fallback with provided assessor")
        else:
            logger.warning("No assessor provided for SystemAwareMerge fallback - mutation will not be available")

        logger.debug("Reset SystemAwareMerge for new compilation with ReflectivePromptMutation fallback")

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
