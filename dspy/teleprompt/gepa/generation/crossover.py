"""System-Aware Merge implementation from GEPA paper Algorithm 4."""

import logging
from typing import List, Optional, Tuple
import dspy
from dspy.teleprompt.utils import get_signature, set_signature
from .generator import Generator
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns
from ..evaluation.trace_collector import EnhancedTraceCollector
from ..utils.ancestry_traversal import find_common_ancestors, is_ancestor_of
from ..utils.signature_complexity import is_signature_more_complex
from ..utils.merge_history import MergeHistoryTracker

logger = logging.getLogger(__name__)


def compare_strict(sig_a, sig_i, sig_j, parent1, parent2):
    """Strict comparison function implementing the GEPA paper's DESIRABLE logic.

    Args:
        sig_a: Ancestor signature
        sig_i: First parent signature
        sig_j: Second parent signature
        parent1: First parent candidate (for performance comparison)
        parent2: Second parent candidate (for performance comparison)

    Returns:
        Selected signature if desirable divergence exists, None otherwise
    """
    # Get the string prompt representations (e.g., "input1, input2 -> output1, output2")
    π_a = sig_a.signature
    π_i = sig_i.signature
    π_j = sig_j.signature

    # Check the DESIRABLE conditions from the paper and return the beneficial signature
    # Condition 1: (πa = πi and πj ≠ πi) → use parent2's innovation
    if π_a == π_i and π_j != π_i:
        return sig_j

    # Condition 2: (πa = πj and πi ≠ πj) → use parent1's innovation
    if π_a == π_j and π_i != π_j:
        return sig_i

    # If all three are different → pick from best performer
    if π_i != π_j and π_i != π_a and π_j != π_a:
        best_parent = parent1 if parent1.average_task_score() > parent2.average_task_score() else parent2
        return sig_i if best_parent == parent1 else sig_j

    return None  # No beneficial divergence


def desirable_generator(compare_func):
    """Generator that creates a desirable function with custom comparison logic.

    Args:
        compare_func: Function that takes (sig_a, sig_i, sig_j, parent1, parent2) and returns signature or None

    Returns:
        A desirable function that returns list of selected signatures per module
    """
    def parametric_desirable(ancestor: "Candidate", parent1: "Candidate", parent2: "Candidate"):
        """Parametric DESIRABLE function from the GEPA paper.

        Implements the algorithm:
        for module m = 1 to |M| do
            selected_sig = compare_func(sig_a, sig_i, sig_j, parent1, parent2)
            if selected_sig is not None:
                collect selected_sig for module m
            end if
        end for
        return list of selected signatures (empty if no desirable patterns)

        Args:
            ancestor: Common ancestor candidate to use as merge base
            parent1: First parent candidate
            parent2: Second parent candidate

        Returns:
            List of (module_index, selected_signature) tuples for modules with desirable patterns
        """
        try:
            from dspy.teleprompt.utils import get_signature

            # Get predictors for all three candidates
            ancestor_predictors = ancestor.module.predictors()
            parent1_predictors = parent1.module.predictors()
            parent2_predictors = parent2.module.predictors()

            if not ancestor_predictors:
                return []  # No predictors to compare

            selected_signatures = []

            # Check each module (predictor) for desirable patterns
            max_modules = max(len(ancestor_predictors), len(parent1_predictors), len(parent2_predictors))

            for m in range(max_modules):
                # Get signatures for module m
                sig_a = get_signature(ancestor_predictors[m]) if m < len(ancestor_predictors) else None
                sig_i = get_signature(parent1_predictors[m]) if m < len(parent1_predictors) else None
                sig_j = get_signature(parent2_predictors[m]) if m < len(parent2_predictors) else None

                # Skip if any signature is missing
                if sig_a is None or sig_i is None or sig_j is None:
                    continue

                # Use the provided comparison function
                selected_sig = compare_func(sig_a, sig_i, sig_j, parent1, parent2)
                if selected_sig is not None:
                    selected_signatures.append((m, selected_sig))
                    logger.debug(f"DESIRABLE: Module {m} shows desirable pattern")

            # Return selected signatures
            if selected_signatures:
                logger.debug(f"DESIRABLE: Found {len(selected_signatures)} desirable modules")
            else:
                logger.debug("DESIRABLE: No divergence patterns found")

            return selected_signatures

        except Exception as e:
            logger.warning(f"DESIRABLE function failed: {e}")
            return []  # Default to no merge on error

    return parametric_desirable


# Default desirable function using strict comparison
desirable = desirable_generator(compare_strict)


class SystemAwareMerge(Generator):
    """System-Aware Merge generator implementing Algorithm 4 from GEPA paper.

    Implements sophisticated module-wise merging that:
    1. Checks for complementary evolution between parents using DESIRABLE function
    2. Uses ancestry-aware selection to avoid inbreeding
    3. Performs intelligent module-wise combination based on performance
    """

    def __init__(self,
                 merge_rate: float = 0.7,
                 population_size: int = 10,
                 feedback_collector: Optional[EnhancedTraceCollector] = None):
        self.merge_rate = merge_rate
        self.population_size = population_size
        self.feedback_collector = feedback_collector or EnhancedTraceCollector()
        # Training data will be set during compilation
        self.feedback_data: List[dspy.Example] = []
        # Merge history tracking as per Algorithm 4
        self.merge_history = MergeHistoryTracker()

    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate new candidates using System-Aware Merge (Algorithm 4)."""
        if parents.size() < 2:
            return NewBorns()

        try:
            # Step 1: Stochastic selection of two parent candidates (Algorithm 2 line 14)
            selected_parents = parents.sample_stochastic(2)
            if selected_parents.size() < 2:
                return NewBorns()

            parent_list = list(selected_parents)
            parent1, parent2 = parent_list[0], parent_list[1]

            # Step 2: Get ancestors Ai ← GET_ANCESTORS(i, A), Aj ← GET_ANCESTORS(j, A)
            # Step 4: For each common ancestor a ∈ Ai ∩ Aj do
            common_ancestors = find_common_ancestors(parent1, parent2)

            # Step 3: Skip direct ancestry: if i ∈ Aj or j ∈ Ai then continue
            if is_ancestor_of(parent1, parent2) or is_ancestor_of(parent2, parent1):
                return NewBorns()

            # Try each common ancestor until one merge succeeds
            for ancestor in common_ancestors:
                # Step 5: Check merge history: if (i, j, a) tried before then continue
                if self.merge_history.has_been_attempted(parent1, parent2, ancestor):
                    continue

                # Step 6: Check signature complexity: if S[a] > min(S[i], S[j]) then continue
                if self._ancestor_too_complex(ancestor, parent1, parent2):
                    self.merge_history.record_attempt(parent1, parent2, ancestor, parents.iteration,
                                                    successful=False, failure_reason="ancestor_too_complex")
                    continue

                # Step 7: Check desirability: if not DESIRABLE(a, i, j, P) then continue
                selected_signatures = desirable(ancestor, parent1, parent2)
                if not selected_signatures:
                    self.merge_history.record_attempt(parent1, parent2, ancestor, parents.iteration,
                                                    successful=False, failure_reason="not_desirable")
                    continue

                # Step 8-11: Create merged candidate with pre-selected signatures
                child_candidate = self._create_merged_candidate(ancestor, parent1, parent2, parents.iteration, selected_signatures)

                if child_candidate is not None:
                    # Record successful merge
                    self.merge_history.record_attempt(parent1, parent2, ancestor, parents.iteration, successful=True)
                    return NewBorns(child_candidate)
                else:
                    self.merge_history.record_attempt(parent1, parent2, ancestor, parents.iteration,
                                                    successful=False, failure_reason="merge_failed")

            logger.debug(f"No successful merge found for parents despite {len(common_ancestors)} common ancestors")

        except Exception as e:
            logger.warning(f"System-Aware Merge failed: {e}")

        return NewBorns()


    def _ancestor_too_complex(self, ancestor: Candidate, parent1: Candidate, parent2: Candidate) -> bool:
        """Check if ancestor signature is too complex compared to parents.

        Implements: S[a] > min(S[i], S[j]) from Algorithm 4

        Args:
            ancestor: Ancestor candidate
            parent1: First parent candidate
            parent2: Second parent candidate

        Returns:
            True if ancestor is too complex (should be skipped)
        """
        try:
            # Get signatures for all three candidates
            ancestor_predictors = ancestor.module.predictors()
            parent1_predictors = parent1.module.predictors()
            parent2_predictors = parent2.module.predictors()

            if not ancestor_predictors:
                return False  # No predictors to compare

            # Check complexity for each predictor position
            max_predictors = max(len(ancestor_predictors), len(parent1_predictors), len(parent2_predictors))

            for i in range(max_predictors):
                # Get signatures (use None if predictor doesn't exist at this position)
                ancestor_sig = get_signature(ancestor_predictors[i]) if i < len(ancestor_predictors) else None
                parent1_sig = get_signature(parent1_predictors[i]) if i < len(parent1_predictors) else None
                parent2_sig = get_signature(parent2_predictors[i]) if i < len(parent2_predictors) else None

                # Skip if any signature is missing
                if not all([ancestor_sig, parent1_sig, parent2_sig]):
                    continue

                # Check if ancestor signature is more complex than both parents
                if is_signature_more_complex(ancestor_sig, parent1_sig, parent2_sig):
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking ancestor complexity: {e}")
            return False  # Default to allowing ancestor

    def _create_merged_candidate(self, ancestor: Candidate, parent1: Candidate, parent2: Candidate, iteration: int, selected_signatures: List[Tuple[int, any]]) -> Optional[Candidate]:
        """Create merged candidate using Algorithm 4 steps 8-11.

        Args:
            ancestor: Common ancestor to use as base
            parent1: First parent candidate
            parent2: Second parent candidate
            iteration: Current iteration number
            selected_signatures: List of (module_index, signature) tuples from desirable function

        Returns:
            New merged candidate or None if merge failed
        """
        try:
            # Step 8: Create merged candidate Φ' ← copy of P[a]
            child_module = ancestor.module.deepcopy()
            child_predictors = child_module.predictors()

            # Step 9-10: Apply pre-selected signatures from desirable function
            for module_idx, selected_signature in selected_signatures:
                try:
                    if module_idx < len(child_predictors):
                        set_signature(child_predictors[module_idx], selected_signature)
                        logger.debug(f"Applied selected signature to module {module_idx}")
                except Exception as e:
                    logger.warning(f"Failed to apply signature to module {module_idx}: {e}")
                    continue

            # Step 11: Create new candidate with 3-way lineage tracking
            child_candidate = Candidate(
                module=child_module,
                parents=[parent1, parent2, ancestor],  # 3-way lineage as per paper
                generation_number=iteration
            )

            return child_candidate

        except Exception as e:
            logger.warning(f"Merged candidate creation failed: {e}")
            return None

    def get_merge_statistics(self) -> dict:
        """Get statistics about merge attempts for debugging/monitoring.

        Returns:
            Dictionary with merge attempt statistics
        """
        return self.merge_history.get_merge_statistics()

    def clear_merge_history(self) -> None:
        """Clear merge history (useful for testing or reset)."""
        self.merge_history.clear_history()

    def start_compilation(self, student: dspy.Module, training_data: List[dspy.Example]) -> None:
        """Prepare generator with training dataset when compilation begins."""
        self.feedback_data = training_data
