"""Configuration-driven ReflectivePromptMutation using DSPy's native systems."""

import logging
from typing import Any, List, Optional
import random

import dspy
from .generator import Generator
from .reflection_strategy import ReflectionStrategy, GEPAReflection
from .evolvable_module import EvolvableModule
from .config import ReflectiveMutationConfig, ModuleSelectionStrategy
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns

logger = logging.getLogger(__name__)


class ReflectivePromptMutation(Generator):
    """Configuration-driven GEPA reflective mutation using DSPy's native systems.
    
    This version uses ReflectiveMutationConfig to make the generator's behavior
    transparent, tunable, and easily experimentable without code changes.
    
    Reuses DSPy's built-in capabilities:
    - Module.deepcopy() for creating mutated copies
    - dspy.context(trace=[]) for trace collection
    - Native predictors() and signature system
    """
    
    def __init__(self, 
                 feedback_provider = None,
                 reflection_strategy: Optional[ReflectionStrategy] = None,
                 reflection_lm: Optional[Any] = None,
                 config: Optional[ReflectiveMutationConfig] = None,
                 **legacy_kwargs):
        """Initialize ReflectivePromptMutation with configuration-driven approach.
        
        Args:
            feedback_provider: REQUIRED (unless provided via config) - Encapsulates metric μ and feedback function μf
            reflection_strategy: Strategy for generating improved instructions (default: GEPAReflection)
            reflection_lm: Language model for reflection (passed to strategy)
            config: ReflectiveMutationConfig object (recommended approach for new code)
            **legacy_kwargs: Backward compatibility for old parameters (minibatch_size, module_selection)
        """
        # Determine if we're in config mode or legacy mode
        # Config mode only if config object is explicitly provided
        config_mode = config is not None
        
        # Initialize with config or create default
        self.config = config or ReflectiveMutationConfig()
        
        # Handle legacy parameters and config precedence
        self._setup_from_config_and_legacy(
            feedback_provider, reflection_strategy, reflection_lm, legacy_kwargs, config_mode
        )
        
        # Validate required components
        if self.feedback_provider is None:
            raise ValueError("ReflectivePromptMutation requires a FeedbackProvider")
        
        # Track state
        self.feedback_data = []
        self.next_module_idx = 0
        self.mutation_count = 0
        self.module_performance_history = {}  # For worst-performing selection
        
        if self.config.enable_detailed_logging:
            logger.setLevel(logging.DEBUG)
    
    def _setup_from_config_and_legacy(self, feedback_provider, reflection_strategy, 
                                     reflection_lm, legacy_kwargs, config_mode):
        """Setup components from config with legacy parameter fallback."""
        if config_mode:
            # Config mode: Config values take precedence, legacy only fills gaps
            # Only use config values if they're explicitly set
            if hasattr(self.config, 'feedback_provider') and self.config.feedback_provider is not None:
                self.feedback_provider = self.config.feedback_provider
            else:
                self.feedback_provider = feedback_provider
            
            if hasattr(self.config, 'reflection_strategy') and self.config.reflection_strategy is not None:
                self.reflection_strategy = self.config.reflection_strategy
            else:
                self.reflection_strategy = reflection_strategy or GEPAReflection()
            
            # In config mode, legacy kwargs are ignored for config-managed parameters
        else:
            # Legacy mode: Use legacy parameters and update config accordingly
            self.feedback_provider = feedback_provider
            self.reflection_strategy = reflection_strategy or GEPAReflection()
            
            # Handle legacy parameters by updating config
            if 'minibatch_size' in legacy_kwargs:
                self.config.minibatch_size = legacy_kwargs['minibatch_size']
            
            if 'module_selection' in legacy_kwargs:
                # Map legacy string values to enum
                legacy_selection = legacy_kwargs['module_selection']
                if legacy_selection == "round_robin":
                    self.config.module_selection_strategy = ModuleSelectionStrategy.ROUND_ROBIN
                elif legacy_selection == "random":
                    self.config.module_selection_strategy = ModuleSelectionStrategy.RANDOM
        
        # Enhanced feedback function (always from config)
        self.enhanced_feedback_function = self.config.enhanced_feedback_function
        
        # Reflection LM (not managed by config currently)
        self.reflection_lm = reflection_lm
    
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
            
            module_idx = self._select_target_module(len(predictors), predictors)
            
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
    
    def _select_target_module(self, num_modules: int, predictors: List = None) -> int:
        """Select module to mutate based on configured strategy."""
        strategy = self.config.module_selection_strategy
        
        if strategy == ModuleSelectionStrategy.ROUND_ROBIN:
            # Paper-compliant round-robin
            module_idx = self.next_module_idx % num_modules
            self.next_module_idx = (self.next_module_idx + 1) % num_modules
            return module_idx
            
        elif strategy == ModuleSelectionStrategy.RANDOM:
            return random.randint(0, num_modules - 1)
            
        elif strategy == ModuleSelectionStrategy.WORST_PERFORMING:
            # Select module with worst recent performance
            return self._select_worst_performing_module(num_modules)
            
        elif strategy == ModuleSelectionStrategy.ALL:
            # For "all" strategy, we select one at random (caller can iterate)
            return random.randint(0, num_modules - 1)
            
        else:
            raise ValueError(f"Unknown module selection strategy: {strategy}")
    
    def _select_worst_performing_module(self, num_modules: int) -> int:
        """Select the worst-performing module based on historical performance."""
        if not self.module_performance_history:
            # No history yet, use round-robin
            return self.next_module_idx % num_modules
        
        # Find module with lowest average performance
        worst_module = 0
        worst_score = float('inf')
        
        for module_idx in range(num_modules):
            if module_idx in self.module_performance_history:
                scores = self.module_performance_history[module_idx]
                avg_score = sum(scores) / len(scores)
                if avg_score < worst_score:
                    worst_score = avg_score
                    worst_module = module_idx
            else:
                # Modules without history are considered worst
                return module_idx
        
        return worst_module
    
    def _sample_minibatch(self) -> List[dspy.Example]:
        """Sample minibatch from feedback data (paper compliant)."""
        if not self.feedback_data:
            return []
        
        batch_size = min(self.config.minibatch_size, len(self.feedback_data))
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
            
            # Update performance history for worst-performing selection
            if module_idx not in self.module_performance_history:
                self.module_performance_history[module_idx] = []
            self.module_performance_history[module_idx].append(child_avg)
            # Keep only recent history
            if len(self.module_performance_history[module_idx]) > 10:
                self.module_performance_history[module_idx] = self.module_performance_history[module_idx][-10:]
            
            # Check if child meets minimum score threshold
            if child_avg < self.config.minimum_score_threshold:
                if self.config.enable_detailed_logging:
                    logger.debug(f"Child score {child_avg:.3f} below threshold {self.config.minimum_score_threshold}")
                return False
            
            # Check improvement requirements
            if not self.config.requires_improvement:
                # Accept any mutation that meets minimum threshold
                improved = True
            else:
                # Require meaningful improvement
                improvement = child_avg - parent_avg
                improved = improvement >= self.config.improvement_threshold
            
            if self.config.enable_detailed_logging:
                logger.debug(f"Validation: parent={parent_avg:.3f}, child={child_avg:.3f}, "
                           f"improvement={child_avg-parent_avg:.3f}, meets_criteria={improved}")
            
            return improved
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return self.config.preserve_original_on_failure == False  # If False, try anyway