"""EvolvableModule using DSPy's native capabilities with PromptMutator separation."""

import logging
from typing import Dict, List, Optional, Any

import dspy
from dspy import Module

from .prompt_mutator import PromptMutator, ReflectivePromptMutator
from .reflection_strategy import GEPAReflection
from ..evaluation.feedback import FeedbackResult

logger = logging.getLogger(__name__)


class EvolvableModule(Module):
    """DSPy Module with evolution capabilities using native DSPy systems.
    
    Handles execution and feedback collection using DSPy's built-in systems,
    while delegating mutation logic to configurable PromptMutator strategies.
    """
    
    def __init__(self, 
                 base_module: Optional[Module] = None,
                 prompt_mutator: Optional[PromptMutator] = None,
                 **kwargs):
        """Initialize EvolvableModule.
        
        Args:
            base_module: Existing DSPy module to wrap (copies all attributes)
            prompt_mutator: Strategy for mutating modules based on feedback
            **kwargs: Additional arguments passed to Module.__init__
        """
        super().__init__(**kwargs)
        
        # Copy from base module if provided (using DSPy's native systems)
        if base_module:
            self._copy_from(base_module)
        
        # Configure mutation strategy (default to reflective)
        self.prompt_mutator = prompt_mutator or ReflectivePromptMutator(GEPAReflection())
        
        # Track evolution
        self._generation_number = 0
        self._evolution_history: List[Dict] = []
    
    def _copy_from(self, other_module: Module):
        """Copy all attributes from another DSPy module."""
        try:
            # Copy all non-private attributes
            for attr_name in dir(other_module):
                if (not attr_name.startswith('_') and 
                    hasattr(other_module, attr_name) and
                    attr_name not in ['reflection_strategy', 'reflection_lm']):  # Skip our own attrs
                    
                    try:
                        attr_value = getattr(other_module, attr_name)
                        # Deep copy predictors to avoid shared state
                        if hasattr(attr_value, 'deepcopy'):
                            setattr(self, attr_name, attr_value.deepcopy())
                        else:
                            setattr(self, attr_name, attr_value)
                    except (AttributeError, TypeError):
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to copy some attributes: {e}")
    
    def collect_traces_and_evaluate(self, 
                                  examples: List[dspy.Example], 
                                  feedback_provider,
                                  target_module_idx: int = 0) -> FeedbackResult:
        """Execute on examples and collect traces using DSPy's native system.
        
        Args:
            examples: Examples to execute on
            feedback_provider: Provider for evaluation and feedback
            target_module_idx: Index of module to focus analysis on
            
        Returns:
            FeedbackResult with scores, diagnostics, and traces
        """
        scores = []
        diagnostics = []
        traces = []
        
        for example in examples:
            try:
                # Use DSPy's native trace collection
                with dspy.context(trace=[]):
                    prediction = self(**example.inputs())
                    trace = dspy.settings.trace.copy() if hasattr(dspy.settings, 'trace') else []
                
                # Use feedback provider for evaluation
                score, diagnostic = feedback_provider.evaluate(example, prediction, trace, target_module_idx)
                
                scores.append(score)
                diagnostics.append(diagnostic)
                traces.append(trace)
                
            except Exception as e:
                scores.append(0.0)
                diagnostics.append(f"ERROR: {str(e)}")
                traces.append([])
        
        return FeedbackResult(scores=scores, diagnostics=diagnostics, traces=traces)
    
    def evolve(self, 
               feedback: FeedbackResult, 
               target_module_idx: int = 0,
               prompt_mutator: Optional[PromptMutator] = None) -> "EvolvableModule":
        """Create evolved copy using DSPy's native systems and PromptMutator.
        
        Args:
            feedback: Feedback from minibatch execution
            target_module_idx: Index of predictor to mutate
            prompt_mutator: Optional custom mutator (uses self.prompt_mutator if None)
            
        Returns:
            New EvolvableModule with mutations applied
        """
        try:
            # Use provided mutator or default to configured one
            mutator = prompt_mutator or self.prompt_mutator
            
            # Step 1: Apply mutation using PromptMutator (handles DSPy deepcopy internally)
            mutated_module = mutator.mutate(self, feedback, target_module_idx)
            
            # Step 2: Wrap mutated module as EvolvableModule
            evolved_copy = EvolvableModule(
                base_module=mutated_module,
                prompt_mutator=self.prompt_mutator
            )
            
            # Step 3: Update evolution metadata
            evolved_copy._generation_number = self._generation_number + 1
            evolved_copy._evolution_history = self._evolution_history + [{
                'generation': evolved_copy._generation_number,
                'target_module': target_module_idx,
                'avg_score': sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0,
                'mutator_type': type(mutator).__name__,
                'mutation_applied': True
            }]
            
            logger.debug(f"Evolved to generation {evolved_copy._generation_number} using {type(mutator).__name__}")
            return evolved_copy
            
        except Exception as e:
            logger.warning(f"Evolution failed: {e}")
            # Return unchanged copy on failure using DSPy's deepcopy
            return EvolvableModule(
                base_module=self.deepcopy(),
                prompt_mutator=self.prompt_mutator
            )
    
    def with_mutator(self, mutator: PromptMutator) -> "EvolvableModule":
        """Create new EvolvableModule with different PromptMutator.
        
        Args:
            mutator: New prompt mutation strategy
            
        Returns:
            New EvolvableModule with updated mutator
        """
        return EvolvableModule(
            base_module=self.deepcopy(),  # DSPy's native deepcopy
            prompt_mutator=mutator
        )
    
    @property
    def generation_number(self) -> int:
        """Get generation number."""
        return self._generation_number
    
    @property
    def evolution_history(self) -> List[Dict]:
        """Get evolution history."""
        return self._evolution_history.copy()
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution summary."""
        return {
            "generation_number": self._generation_number,
            "mutations_applied": len(self._evolution_history),
            "reflection_strategy": type(self.reflection_strategy).__name__,
            "avg_performance": (
                sum(entry['avg_score'] for entry in self._evolution_history) / 
                len(self._evolution_history)
            ) if self._evolution_history else 0.0
        }