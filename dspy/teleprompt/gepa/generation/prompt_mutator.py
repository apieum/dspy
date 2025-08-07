"""Prompt mutation strategies using DSPy's native systems."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import dspy
from dspy import Module
from dspy.signatures.signature import make_signature

from .reflection_strategy import ReflectionStrategy, GEPAReflection
from ..evaluation.feedback import FeedbackResult

logger = logging.getLogger(__name__)


class PromptMutator(ABC):
    """Protocol for mutating DSPy modules based on feedback.
    
    Uses DSPy's native systems (deepcopy, predictors, signatures)
    for reliable and efficient mutation.
    """
    
    @abstractmethod
    def mutate(self, module: Module, feedback: FeedbackResult, target_module_idx: int = 0) -> Module:
        """Create a mutated copy of the module based on feedback.
        
        Args:
            module: DSPy module to mutate
            feedback: Feedback from minibatch execution
            target_module_idx: Index of predictor to mutate
            
        Returns:
            New module with mutations applied (original unchanged)
        """
        pass


class ReflectivePromptMutator(PromptMutator):
    """GEPA's reflective prompt mutation using DSPy native systems.
    
    Uses reflection strategy to analyze feedback and improve instructions,
    while leveraging DSPy's built-in module and signature systems.
    """
    
    def __init__(self, 
                 reflection_strategy: Optional[ReflectionStrategy] = None,
                 reflection_lm: Optional[Any] = None):
        """Initialize reflective mutator.
        
        Args:
            reflection_strategy: Strategy for generating improved instructions
            reflection_lm: Language model for reflection (passed to strategy)
        """
        self.reflection_strategy = reflection_strategy or GEPAReflection()
        self.reflection_lm = reflection_lm
        self.mutation_count = 0
    
    def mutate(self, module: Module, feedback: FeedbackResult, target_module_idx: int = 0) -> Module:
        """Mutate module using reflective prompt improvement with DSPy native systems."""
        try:
            # Use DSPy's native deepcopy for safe mutation
            mutated_module = module.deepcopy()
            
            # Use DSPy's native predictors() method
            predictors = mutated_module.predictors()
            if not predictors or target_module_idx >= len(predictors):
                logger.warning("No valid predictor to mutate")
                return mutated_module
            
            target_predictor = predictors[target_module_idx]
            
            # Get current instruction from DSPy signature
            current_instruction = target_predictor.signature.instructions or "Answer the question."
            
            # Format feedback for reflection
            formatted_examples = self._format_feedback_for_reflection(feedback, target_module_idx)
            
            # Use reflection strategy to generate improved instruction
            improved_instruction = self.reflection_strategy.reflect(
                current_instruction=current_instruction,
                formatted_examples=formatted_examples,
                prompt_model=self.reflection_lm
            )
            
            # Apply improved instruction using DSPy's signature system
            self._update_predictor_instruction(target_predictor, improved_instruction)
            
            # Track mutation
            self.mutation_count += 1
            logger.debug(f"Applied reflective mutation #{self.mutation_count}: {current_instruction[:50]}... -> {improved_instruction[:50]}...")
            
            return mutated_module
            
        except Exception as e:
            logger.warning(f"Reflective mutation failed: {e}")
            return module.deepcopy()  # Return copy of original on failure
    
    def _format_feedback_for_reflection(self, feedback: FeedbackResult, target_module_idx: int) -> str:
        """Format feedback using DSPy's native trace format."""
        if not feedback.scores or not feedback.diagnostics:
            return "No feedback available."
        
        formatted_parts = []
        for i, (score, diagnostic) in enumerate(zip(feedback.scores, feedback.diagnostics)):
            # Extract trace information using DSPy's standard trace format
            trace_info = "No trace"
            if (feedback.traces and i < len(feedback.traces) and 
                feedback.traces[i] and target_module_idx < len(feedback.traces[i])):
                
                # DSPy trace format: (predictor, inputs, outputs)
                predictor, inputs, outputs = feedback.traces[i][target_module_idx]
                
                # Format inputs/outputs from DSPy trace
                input_str = ", ".join([f"{k}: {str(v)[:30]}..." if len(str(v)) > 30 else f"{k}: {v}" 
                                     for k, v in inputs.items()])
                output_str = ", ".join([f"{k}: {str(v)[:30]}..." if len(str(v)) > 30 else f"{k}: {v}"
                                      for k, v in outputs.items()])
                
                trace_info = f"Input: {input_str} → Output: {output_str}"
            
            example_text = f"""Example {i+1}:
Score: {score:.2f}
Feedback: {diagnostic}  
Execution: {trace_info}"""
            formatted_parts.append(example_text)
        
        return "\n\n".join(formatted_parts)
    
    def _update_predictor_instruction(self, predictor: Any, improved_instruction: str):
        """Update predictor's signature instruction using DSPy's native signature system."""
        try:
            current_signature = predictor.signature
            
            # Use DSPy's native field access methods
            input_fields = list(current_signature.input_fields.keys())
            output_fields = list(current_signature.output_fields.keys())
            
            # Create new signature using DSPy's make_signature
            field_signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
            new_signature = make_signature(field_signature, improved_instruction)
            
            # Update predictor's signature (DSPy native)
            predictor.signature = new_signature
            
        except Exception as e:
            logger.warning(f"Failed to update predictor instruction using DSPy systems: {e}")


class SimplePromptMutator(PromptMutator):
    """Simple template-based prompt mutation using DSPy native systems.
    
    Applies predefined templates to instructions without LLM reflection,
    useful for baseline comparisons or when reflection is not desired.
    """
    
    def __init__(self, mutation_templates: List[str] = None):
        """Initialize simple mutator.
        
        Args:
            mutation_templates: Templates for instruction mutations
        """
        self.mutation_templates = mutation_templates or [
            "Think step by step. {}",
            "Be precise and accurate. {}",
            "Consider all aspects carefully. {}",
            "Analyze thoroughly. {}"
        ]
        self.mutation_count = 0
    
    def mutate(self, module: Module, feedback: FeedbackResult, target_module_idx: int = 0) -> Module:
        """Apply template-based mutations using DSPy native systems."""
        try:
            # Use DSPy's native deepcopy
            mutated_module = module.deepcopy()
            
            # Use DSPy's native predictors() method
            predictors = mutated_module.predictors()
            if not predictors or target_module_idx >= len(predictors):
                return mutated_module
            
            target_predictor = predictors[target_module_idx]
            
            # Select template based on performance
            template_idx = self._select_template(feedback)
            template = self.mutation_templates[template_idx]
            
            # Apply template to current instruction
            current_instruction = target_predictor.signature.instructions or "Answer the question."
            improved_instruction = template.format(current_instruction)
            
            # Update using DSPy's signature system
            self._update_predictor_instruction(target_predictor, improved_instruction)
            
            self.mutation_count += 1
            logger.debug(f"Applied template mutation #{self.mutation_count}: {template}")
            
            return mutated_module
            
        except Exception as e:
            logger.warning(f"Simple mutation failed: {e}")
            return module.deepcopy()
    
    def _select_template(self, feedback: FeedbackResult) -> int:
        """Select template based on feedback performance."""
        if not feedback.scores:
            return 0
        
        avg_score = sum(feedback.scores) / len(feedback.scores)
        
        if avg_score < 0.3:
            return 0  # "Think step by step"
        elif avg_score < 0.6:
            return 1  # "Be precise and accurate"
        elif avg_score < 0.8:
            return 2  # "Consider all aspects carefully"
        else:
            return 3  # "Analyze thoroughly"
    
    def _update_predictor_instruction(self, predictor: Any, improved_instruction: str):
        """Update predictor using DSPy's native signature system."""
        try:
            current_signature = predictor.signature
            input_fields = list(current_signature.input_fields.keys())
            output_fields = list(current_signature.output_fields.keys())
            field_signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
            new_signature = make_signature(field_signature, improved_instruction)
            predictor.signature = new_signature
        except Exception as e:
            logger.warning(f"Failed to update predictor instruction: {e}")


class NoOpMutator(PromptMutator):
    """No-operation mutator for baseline comparisons.
    
    Returns unchanged copy using DSPy's native deepcopy,
    useful for testing the impact of mutation vs no mutation.
    """
    
    def mutate(self, module: Module, feedback: FeedbackResult, target_module_idx: int = 0) -> Module:
        """Return unchanged copy using DSPy's native deepcopy."""
        return module.deepcopy()  # Still use DSPy's native copying