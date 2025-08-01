"""Crossover-based candidate generation implementing GEPA+Merge."""

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import dspy
from dspy.teleprompt.utils import get_signature

from ..data.candidate_pool import CandidatePool, CandidateLineage
from .base import CandidateGenerator

logger = logging.getLogger(__name__)


class CrossoverGenerator(CandidateGenerator):
    """Intelligent crossover generator implementing GEPA+Merge.
    
    Implements the crossover strategy from GEPA paper, combining
    complementary instructions from different evolutionary lineages
    using intelligent semantic analysis.
    """
    
    def __init__(self, frequency: int = 5, min_candidates: int = 3):
        """Initialize crossover generator.
        
        Args:
            frequency: Attempt crossover every N iterations
            min_candidates: Minimum candidates required to attempt crossover
        """
        self.frequency = frequency
        self.min_candidates = min_candidates
    
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[dspy.Example], 
                          iteration: int) -> List[dspy.Module]:
        """Generate new candidate through intelligent crossover.
        
        Follows paper's GEPA+Merge approach:
        1. Check if crossover should be attempted (frequency, min candidates)
        2. Select parents from different lineages
        3. Perform intelligent instruction merging
        4. Return merged candidate
        """
        # Check preconditions for crossover
        if (iteration < 3 or 
            iteration % self.frequency != 0 or 
            candidate_pool.size() < self.min_candidates):
            return []
        
        try:
            # Select parents for crossover
            parent_indices = self._select_crossover_parents(candidate_pool)
            if parent_indices is None:
                return []
            
            parent1_idx, parent2_idx = parent_indices
            parent1 = candidate_pool.get_candidate(parent1_idx)
            parent2 = candidate_pool.get_candidate(parent2_idx)
            
            if parent1 is None or parent2 is None:
                return []
            
            # Perform intelligent crossover
            merged_candidate = self._merge_candidates(parent1, parent2)
            
            if merged_candidate is None:
                return []
            
            logger.info(f"Created crossover candidate from parents {parent1_idx} and {parent2_idx}")
            return [merged_candidate]
            
        except Exception as e:
            logger.warning(f"Crossover generation failed: {e}")
            return []
    
    def _select_crossover_parents(self, candidate_pool: CandidatePool) -> Optional[Tuple[int, int]]:
        """Select optimal parents for crossover from different lineages."""
        candidates = candidate_pool.get_candidates()
        scores = candidate_pool.get_scores()
        lineages = candidate_pool.get_lineages()
        
        if len(candidates) < 2:
            return None
        
        # Group candidates by lineage root
        lineage_groups = defaultdict(list)
        for idx in range(len(candidates)):
            lineage = lineages.get(idx)
            if lineage:
                root_ancestor = self._find_root_ancestor(lineage, lineages)
                lineage_groups[root_ancestor].append(idx)
            else:
                lineage_groups[idx].append(idx)  # Original candidate
        
        if len(lineage_groups) < 2:
            return None
        
        # Find best candidates from different lineages
        best_from_lineages = []
        for root, group in lineage_groups.items():
            best_idx = max(group, key=lambda i: scores.compute_average_score(i))
            best_score = scores.compute_average_score(best_idx)
            best_from_lineages.append((best_idx, best_score))
        
        if len(best_from_lineages) < 2:
            return None
        
        # Sort by score and take top 2
        best_from_lineages.sort(key=lambda x: x[1], reverse=True)
        return best_from_lineages[0][0], best_from_lineages[1][0]
    
    def _find_root_ancestor(self, lineage: CandidateLineage, all_lineages: Dict[int, CandidateLineage]) -> int:
        """Find the root ancestor of a lineage."""
        current = lineage
        while current.parent_id is not None and current.parent_id in all_lineages:
            current = all_lineages[current.parent_id]
        return current.candidate_id
    
    def _merge_candidates(self, candidate1: dspy.Module, candidate2: dspy.Module) -> Optional[dspy.Module]:
        """Merge two candidates using intelligent crossover strategies."""
        try:
            # Create base candidate from candidate1
            merged_candidate = candidate1.deepcopy()
            
            # Extract instructions from both candidates
            predictors1 = candidate1.predictors()
            predictors2 = candidate2.predictors()
            
            if len(predictors1) != len(predictors2):
                return None  # Can't merge candidates with different structures
            
            # Intelligent crossover: analyze instruction compatibility
            merged_predictors = merged_candidate.predictors()
            
            for i, (pred1, pred2, merged_pred) in enumerate(zip(predictors1, predictors2, merged_predictors)):
                sig1 = get_signature(pred1)
                sig2 = get_signature(pred2)
                
                inst1 = sig1.instructions or ""
                inst2 = sig2.instructions or ""
                
                if inst1 and inst2 and inst1 != inst2:
                    # Apply intelligent crossover strategy
                    combined_instruction = self._intelligent_crossover(inst1, inst2, i)
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = combined_instruction
                elif inst1 and not inst2:
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = inst1
                elif inst2 and not inst1:
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = inst2
            
            return merged_candidate
            
        except Exception as e:
            logger.warning(f"Candidate merge failed: {e}")
            return None
    
    def _intelligent_crossover(self, inst1: str, inst2: str, module_idx: int) -> str:
        """Apply intelligent crossover strategies to combine instructions."""
        # Strategy 1: Detect complementary instructions
        if self._are_complementary(inst1, inst2):
            return self._complementary_merge(inst1, inst2)
        
        # Strategy 2: Detect adversarial instructions
        elif self._are_adversarial(inst1, inst2):
            return self._adversarial_merge(inst1, inst2)
        
        # Strategy 3: Hybrid merge for general cases
        else:
            return self._hybrid_merge(inst1, inst2, module_idx)
    
    def _are_complementary(self, inst1: str, inst2: str) -> bool:
        """Check if instructions are complementary (non-overlapping guidance)."""
        complementary_pairs = [
            (['accurate', 'precise', 'exact'], ['clear', 'concise', 'brief']),
            (['step', 'systematic', 'methodical'], ['creative', 'innovative', 'flexible']),
            (['specific', 'detailed', 'thorough'], ['efficient', 'quick', 'direct']),
            (['analytical', 'logical', 'reasoning'], ['intuitive', 'practical', 'applied'])
        ]
        
        inst1_lower = inst1.lower()
        inst2_lower = inst2.lower()
        
        for group1, group2 in complementary_pairs:
            has_group1_in_inst1 = any(kw in inst1_lower for kw in group1)
            has_group2_in_inst2 = any(kw in inst2_lower for kw in group2)
            has_group1_in_inst2 = any(kw in inst2_lower for kw in group1)
            has_group2_in_inst1 = any(kw in inst1_lower for kw in group2)
            
            # Complementary if one instruction focuses on group1 and other on group2
            if (has_group1_in_inst1 and has_group2_in_inst2 and 
                not has_group1_in_inst2 and not has_group2_in_inst1):
                return True
                
        return False
    
    def _are_adversarial(self, inst1: str, inst2: str) -> bool:
        """Check if instructions are adversarial (conflicting guidance)."""
        adversarial_pairs = [
            (['brief', 'concise', 'short'], ['detailed', 'comprehensive', 'thorough']),
            (['simple', 'basic', 'straightforward'], ['complex', 'sophisticated', 'advanced']),
            (['fast', 'quick', 'rapid'], ['careful', 'methodical', 'deliberate']),
            (['direct', 'immediate'], ['step-by-step', 'gradual', 'incremental'])
        ]
        
        inst1_lower = inst1.lower()
        inst2_lower = inst2.lower()
        
        for group1, group2 in adversarial_pairs:
            has_group1_in_inst1 = any(kw in inst1_lower for kw in group1)
            has_group2_in_inst2 = any(kw in inst2_lower for kw in group2)
            has_group1_in_inst2 = any(kw in inst2_lower for kw in group1)
            has_group2_in_inst1 = any(kw in inst1_lower for kw in group2)
            
            # Adversarial if instructions contain conflicting directives
            if ((has_group1_in_inst1 and has_group2_in_inst2) or 
                (has_group1_in_inst2 and has_group2_in_inst1)):
                return True
                
        return False
    
    def _complementary_merge(self, inst1: str, inst2: str) -> str:
        """Merge complementary instructions by combining their strengths."""
        return f"{inst1} {inst2}"
    
    def _adversarial_merge(self, inst1: str, inst2: str) -> str:
        """Merge adversarial instructions by finding balanced middle ground.""" 
        return f"{inst1} While being {inst2.lower()}, maintain balance and effectiveness."
    
    def _hybrid_merge(self, inst1: str, inst2: str, module_idx: int) -> str:
        """Hybrid merge strategy for general cases."""
        if module_idx == 0:  # First module - emphasize clarity
            return f"{inst1} Ensure clarity: {inst2.lower()}"
        elif module_idx % 2 == 0:  # Even modules - structured combination
            return f"{inst1} Additionally, {inst2.lower()}"
        else:  # Odd modules - adaptive combination
            return f"Combine approaches: {inst1.lower()} and {inst2.lower()}"