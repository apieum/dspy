"""Simplified ReflectivePromptMutation using DSPy's native systems."""

import logging
from typing import Any, List, Optional
import random

import dspy
from .generator import Generator
from .reflection_strategy import ReflectionStrategy, GEPAReflection
from .evolvable_module import EvolvableModule
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns

logger = logging.getLogger(__name__)


class ReflectivePromptMutation(Generator):
    """GEPA reflective mutation using DSPy's native systems.
    
    This simplified version reuses DSPy's built-in capabilities:
    - Module.deepcopy() for creating mutated copies
    - dspy.context(trace=[]) for trace collection
    - Native predictors() and signature system
    """
    
    def __init__(self, 
                 feedback_provider,
                 reflection_strategy: Optional[ReflectionStrategy] = None,
                 reflection_lm: Optional[Any] = None,
                 minibatch_size: int = 3,
                 module_selection: str = "round_robin"):
        """Initialize ReflectivePromptMutation with DSPy-native approach.
        
        Args:
            feedback_provider: REQUIRED - Encapsulates metric μ and feedback function μf
            reflection_strategy: Strategy for generating improved instructions (default: GEPAReflection)
            reflection_lm: Language model for reflection (passed to strategy)
            minibatch_size: Number of examples for mutation feedback (paper's 'b' parameter)
            module_selection: Strategy for module selection ("round_robin" or "random")
        """
        if feedback_provider is None:
            raise ValueError("ReflectivePromptMutation requires a FeedbackProvider")
        
        self.feedback_provider = feedback_provider
        self.reflection_strategy = reflection_strategy or GEPAReflection()
        self.reflection_lm = reflection_lm
        self.minibatch_size = minibatch_size
        self.module_selection = module_selection
        
        # Track state
        self.feedback_data = []
        self.next_module_idx = 0
        self.mutation_count = 0
    
    def start_compilation(self, student: dspy.Module, 
                         d_feedback: List[dspy.Example], 
                         d_pareto: List[dspy.Example]) -> None:
        """Initialize with feedback dataset for mutation minibatches (GEPA Algorithm 1)."""
        # Generator uses D_feedback for minibatch sampling (not D_pareto)
        self.feedback_data = d_feedback
        self.next_module_idx = 0
        self.mutation_count = 0
    
    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate new candidates using DSPy's native systems."""
        if parents.is_empty() or not self.feedback_data:
            return NewBorns()
        
        try:
            # Step 1: Select parent using DSPy's stochastic selection
            selected_parents = parents.sample_stochastic(1)
            if selected_parents.is_empty():
                return NewBorns()
            
            parent = list(selected_parents)[0]
            
            # Step 2: Wrap as EvolvableModule (reuses DSPy's Module)
            evolvable = self._ensure_evolvable(parent.module)
            
            # Step 3: Select module using DSPy's predictors()
            predictors = evolvable.predictors()  # DSPy native method
            if not predictors:
                return NewBorns()
            
            module_idx = self._select_target_module(len(predictors))
            
            # Step 4: Sample minibatch (paper compliant)
            minibatch = self._sample_minibatch()
            if not minibatch:
                return NewBorns()
            
            # Step 5: Execute and collect feedback using DSPy's trace system
            feedback = evolvable.collect_traces_and_evaluate(
                minibatch, self.feedback_provider, module_idx
            )
            
            # Step 6: Evolve using DSPy's deepcopy and signature system
            child_module = evolvable.evolve(feedback, module_idx)
            
            # Step 7: Validate improvement (paper: Algorithm 1 Lines 13-14)
            if self._validate_improvement(parent, child_module, minibatch, module_idx):
                child_candidate = Candidate(
                    module=child_module,
                    generation_number=parent.generation_number + 1,
                    parents=[parent],
                    creation_metadata={
                        "mutation_type": "reflective_prompt",
                        "target_module": str(module_idx),
                        "reflection_strategy": type(self.reflection_strategy).__name__,
                        "minibatch_size": len(minibatch),
                        "mutation_count": self.mutation_count
                    }
                )
                
                self.mutation_count += 1
                return NewBorns(child_candidate, iteration=parents.iteration)
            else:
                logger.debug("Mutation did not improve performance")
                return NewBorns()
        
        except Exception as e:
            logger.warning(f"Reflective mutation failed: {e}")
            return NewBorns()
    
    def _ensure_evolvable(self, module: dspy.Module) -> EvolvableModule:
        """Wrap DSPy module as EvolvableModule."""
        if isinstance(module, EvolvableModule):
            return module
        
        from .prompt_mutator import ReflectivePromptMutator
        return EvolvableModule(
            base_module=module,
            prompt_mutator=ReflectivePromptMutator(
                reflection_strategy=self.reflection_strategy,
                reflection_lm=self.reflection_lm
            )
        )
    
    def _select_target_module(self, num_modules: int) -> int:
        """Select module to mutate."""
        if self.module_selection == "round_robin":
            # Paper-compliant round-robin
            module_idx = self.next_module_idx % num_modules
            self.next_module_idx = (self.next_module_idx + 1) % num_modules
            return module_idx
        elif self.module_selection == "random":
            return random.randint(0, num_modules - 1)
        else:
            raise ValueError(f"Unknown module selection strategy: {self.module_selection}")
    
    def _sample_minibatch(self) -> List[dspy.Example]:
        """Sample minibatch from feedback data (paper compliant)."""
        if not self.feedback_data:
            return []
        
        batch_size = min(self.minibatch_size, len(self.feedback_data))
        return random.sample(self.feedback_data, batch_size)
    
    def _validate_improvement(self, 
                            parent: Candidate,
                            child_module: EvolvableModule, 
                            minibatch: List[dspy.Example],
                            module_idx: int) -> bool:
        """Validate improvement using DSPy's native execution."""
        try:
            # Execute parent on same minibatch using DSPy's native trace system
            parent_evolvable = self._ensure_evolvable(parent.module)
            parent_feedback = parent_evolvable.collect_traces_and_evaluate(
                minibatch, self.feedback_provider, module_idx
            )
            parent_avg = sum(parent_feedback.scores) / len(parent_feedback.scores) if parent_feedback.scores else 0.0
            
            # Execute child on same minibatch
            child_feedback = child_module.collect_traces_and_evaluate(
                minibatch, self.feedback_provider, module_idx
            )
            child_avg = sum(child_feedback.scores) / len(child_feedback.scores) if child_feedback.scores else 0.0
            
            # Check improvement (Algorithm 1 Line 14)
            improved = child_avg > parent_avg
            
            logger.debug(f"Validation: parent={parent_avg:.3f}, child={child_avg:.3f}, improved={improved}")
            return improved
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return False