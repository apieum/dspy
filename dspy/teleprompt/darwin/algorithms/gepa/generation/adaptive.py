"""GEPA Paper-Compliant Adaptive Generator.

This implements the true GEPA algorithm: mutation as the primary strategy
with opportunistic merging when complementary lineages are detected.
This replaces the artificial split between GEPAMute and GEPAMerge.
"""

import logging
from typing import Optional, TYPE_CHECKING

import dspy
from ....data.cohort import Parents, NewBorns
from ....generation.generator import Generator
from .mutation import ReflectivePromptMutation
from .system_aware_merge import SystemAwareMerge

if TYPE_CHECKING:
    from ....budget import Budget
    from ....config import DarwinConfig

logger = logging.getLogger(__name__)


class GEPAAdaptiveGenerator(Generator):
    """GEPA paper-compliant generator using mutation + opportunistic merging.
    
    This implements the actual GEPA algorithm described in the paper:
    1. Primary strategy: Reflective mutation (improve individual lineages)
    2. Opportunistic strategy: Merge when complementary lineages are detected
    3. SystemAwareMerge already implements desirable() function for complementarity detection
    
    The key insight is that merging should be opportunistic, not the primary strategy.
    """
    
    def __init__(self, 
                 *,
                 config: "DarwinConfig",
                 mutation_generator: Optional[ReflectivePromptMutation] = None,
                 merge_generator: Optional[SystemAwareMerge] = None,
                 feedback_provider=None,
                 feedback_data=None,
                 assessor=None,
                 **kwargs):
        """Initialize adaptive generator with mutation and merge strategies.
        
        Args:
            mutation_generator: ReflectivePromptMutation for primary strategy
            merge_generator: SystemAwareMerge for opportunistic strategy
        """
        super().__init__()
        # Create default generators if not provided
        self.mutation_gen = mutation_generator
        self.merge_gen = merge_generator
        self.feedback_provider = feedback_provider
        self.feedback_data = feedback_data or []
        self.assessor = assessor
        self.config = config
        self.use_merge = config.use_merge
        
        # Track strategy usage for analysis
        self.strategy_stats = {
            "mutation_attempts": 0,
            "merge_attempts": 0, 
            "merge_successes": 0,
            "mutation_successes": 0
        }
        
        # Compilation state
        self.split_strategy = None
        self.verbose = False
    
    def start_compilation(
        self,
        student: dspy.Module,
        dataset_manager=None,
        *,
        feedback_data=None,
        verbose: bool = False,
    ) -> None:
        """Initialize generators with compilation context."""
        self.verbose = verbose
        if feedback_data is not None:
            self.feedback_data = feedback_data
        
        # Initialize mutation generator if not provided
        if self.mutation_gen is None:
            if self.feedback_provider is not None:
                self.mutation_gen = self.config.fallback_mutation(
                    feedback_provider=self.feedback_provider,
                    feedback_data=self.feedback_data,
                    config=self.config,
                )
            else:
                self.publish('mutation_failure', None, {'reason': 'Mutation generator not provided'})
        
        # Initialize merge generator if not provided  
        if self.merge_gen is None:
            self.merge_gen = self.config.crossover(
                assessor=self.assessor,
                config=self.config,
            )
        
        # Initialize child generators (now all use standard interface)
        if self.mutation_gen:
            self.mutation_gen.start_compilation(
                student,
                dataset_manager=dataset_manager,
                feedback_data=self.feedback_data,
                verbose=verbose,
            )
        if self.merge_gen:
            self.merge_gen.start_compilation(
                student,
                dataset_manager=dataset_manager,
                verbose=verbose,
                feedback_data=self.feedback_data,
            )
    
    def finish_compilation(self, result: dspy.Module) -> None:
        """Clean up generators."""
        if self.mutation_gen:
            self.mutation_gen.finish_compilation(result)
        if self.merge_gen:
            self.merge_gen.finish_compilation(result)
        
        # Report strategy usage statistics
        if self.verbose:
            self._publish_strategy_statistics()
    
    def start_iteration(self, iteration: int, cohort, budget) -> None:
        """Start iteration for child generators."""
        if self.mutation_gen:
            self.mutation_gen.start_iteration(iteration, cohort, budget)
        if self.merge_gen:
            self.merge_gen.start_iteration(iteration, cohort, budget)
    
    def finish_iteration(self, iteration: int, cohort, budget) -> None:
        """Finish iteration for child generators."""
        if self.mutation_gen:
            self.mutation_gen.finish_iteration(iteration, cohort, budget)
        if self.merge_gen:
            self.merge_gen.finish_iteration(iteration, cohort, budget)
    
    def generate(self, parents: Parents, budget: Optional["Budget"] = None) -> NewBorns:
        """Generate new candidate using GEPA paper algorithm.
        
        Algorithm:
        1. If we have 2+ parents, try opportunistic merging first
        2. SystemAwareMerge has built-in desirable() logic for complementarity detection
        3. If merge fails or not applicable, fall back to mutation (primary strategy)
        4. Track strategy usage for analysis
        
        Args:
            parents: Parent candidates for generation
            budget: Optional budget tracking
            
        Returns:
            NewBorns cohort with generated candidate
        """
        if parents.is_empty():
            return NewBorns()
        
        # Strategy 1: Try opportunistic merging if we have enough parents
        if self.use_merge and parents.size() >= 2 and self.merge_gen:
            self.strategy_stats["merge_attempts"] += 1
            
            self.publish('mutation_attempt', None, {'attempt': self.strategy_stats["merge_attempts"], 'max_attempts': None})
            
            try:
                merged_child = self.merge_gen.generate(parents, budget)
                
                if not merged_child.is_empty():
                    self.strategy_stats["merge_successes"] += 1
                    # Publish generation success with parent and newborn data
                    self.publish('generate', parents, merged_child)
                    return merged_child
                        
            except Exception as e:
                self.publish('mutation_failure', None, {'reason': f'Merge generation failed: {e}'})
        
        # Strategy 2: Fall back to mutation (primary strategy)
        if self.mutation_gen:
            self.strategy_stats["mutation_attempts"] += 1
            
            self.publish('mutation_attempt', None, {'attempt': self.strategy_stats["mutation_attempts"], 'max_attempts': None})
            
            try:
                mutated_child = self.mutation_gen.generate(parents, budget)
                
                if not mutated_child.is_empty():
                    self.strategy_stats["mutation_successes"] += 1
                    self.publish('mutation_success', None, mutated_child.candidates[0] if mutated_child.candidates else None)
                
                return mutated_child
                
            except Exception as e:
                self.publish('mutation_failure', None, {'reason': f'Mutation generation failed: {e}'})
                return NewBorns()
        else:
            self.publish('mutation_failure', None, {'reason': 'No mutation generator available - adaptive generation failed'})
            return NewBorns()
    
    def set_mutation_generator(self, mutation_gen: ReflectivePromptMutation) -> None:
        """Set the mutation generator (used by factory functions)."""
        self.mutation_gen = mutation_gen
        if self.split_strategy:
            self.mutation_gen.start_compilation(None, feedback_data=self.feedback_data, verbose=self.verbose)
    
    def set_merge_generator(self, merge_gen: SystemAwareMerge) -> None:
        """Set the merge generator (used by factory functions).""" 
        self.merge_gen = merge_gen
        if self.split_strategy:
            self.merge_gen.start_compilation(None, feedback_data=self.feedback_data, verbose=self.verbose)
    
    def _publish_strategy_statistics(self) -> None:
        """Publish statistics about strategy usage via observer notifications."""
        total_attempts = self.strategy_stats["mutation_attempts"] + self.strategy_stats["merge_attempts"]
        
        if total_attempts == 0:
            return
        
        mutation_rate = self.strategy_stats["mutation_attempts"] / total_attempts
        merge_rate = self.strategy_stats["merge_attempts"] / total_attempts
        
        mutation_success = (self.strategy_stats["mutation_successes"] / self.strategy_stats["mutation_attempts"] 
                           if self.strategy_stats["mutation_attempts"] > 0 else 0)
        merge_success = (self.strategy_stats["merge_successes"] / self.strategy_stats["merge_attempts"]
                        if self.strategy_stats["merge_attempts"] > 0 else 0)
        
        # Publish comprehensive strategy statistics
        stats_data = {
            'mutation_rate': mutation_rate,
            'merge_rate': merge_rate,
            'mutation_success_rate': mutation_success,
            'merge_success_rate': merge_success,
            'total_attempts': total_attempts
        }
        self.publish('strategy_statistics', stats_data)
