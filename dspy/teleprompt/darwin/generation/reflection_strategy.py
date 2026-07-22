"""Reflection strategies for prompt mutation in GEPA."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Any

import dspy
from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


class ReflectionStrategy(ABC):
    """Protocol for different reflection approaches to prompt mutation."""
    
    @abstractmethod
    def reflect(self, 
                current_instruction: str,
                formatted_examples: str,
                prompt_model: Optional[Any] = None) -> str:
        """Generate improved instruction based on examples and feedback.
        
        Args:
            current_instruction: Current prompt instruction text
            formatted_examples: Formatted minibatch examples with feedback
            prompt_model: Optional LLM to use for reflection
            
        Returns:
            Improved instruction text
        """
        pass


class GEPAReflectionSignature(dspy.Signature):
    """Propose a new instruction from labeled execution examples and feedback."""
    current_instruction: str = dspy.InputField(
        desc="Current instruction text that needs improvement based on performance feedback"
    )
    formatted_examples: str = dspy.InputField(
        desc="Labeled execution examples containing inputs, expected outputs, actual outputs, scores, and feedback used to diagnose failures and improve the instruction"
    )

    new_instruction: str = dspy.OutputField(
        desc="A complete improved instruction for the assistant, including task-specific knowledge and actionable guidance inferred from the examples and feedback."
    )


class GEPAReflection(ReflectionStrategy):
    """Paper-compliant GEPA reflection using DSPy ChainOfThought.
    
    This is the default reflection strategy implementing the exact
    approach described in the GEPA paper with natural language reflection.
    """
    
    def __init__(self, optimized_reflector=None):
        if optimized_reflector:
            # Use pre-optimized reflector (e.g., via MIPROv2)
            self.reflector = optimized_reflector
        else:
            # Official GEPA uses a single focused instruction-proposal call.
            # Keeping the proposal output narrow leaves more context for the
            # labeled examples and makes the reflection task unambiguous.
            self.reflector = dspy.Predict(GEPAReflectionSignature)
    
    def reflect(self, 
                current_instruction: str,
                formatted_examples: str,
                prompt_model: Optional[Any] = None) -> str:
        """Use DSPy ChainOfThought to reflect on examples and improve instruction."""
        try:
            # Use provided model or default settings
            with dspy.context(lm=prompt_model) if prompt_model else dspy.context():
                reflection_result = self.reflector(
                    current_instruction=current_instruction,
                    formatted_examples=formatted_examples
                )
            
            # Log the reflection process for debugging
            # Return the improved instruction with safety check
            if hasattr(reflection_result, 'new_instruction') and reflection_result.new_instruction:
                return reflection_result.new_instruction
            else:
                logger.warning(f"GEPA reflection returned empty new_instruction, using original instruction")
                return current_instruction
            
        except Exception as e:
            logger.warning(f"GEPA reflection failed: {e}, returning original instruction")
            return current_instruction


def create_optimized_reflection_strategy(reflection_trainset=None, optimizer_type="miprov2"):
    """Create an optimized reflection strategy using DSPy optimizers.
    
    Args:
        reflection_trainset: Training examples for optimizing reflection prompts
        optimizer_type: Type of optimizer to use ("miprov2", "bootstrap", etc.)
    
    Returns:
        GEPAReflection with optimized reflector
    """
    if not reflection_trainset:
        logger.info("No reflection training data provided, using default reflection strategy")
        return GEPAReflection()
    
    try:
        from dspy.teleprompt import MIPROv2, BootstrapFewShot
        
        # Base reflector to optimize
        base_reflector = dspy.Predict(GEPAReflectionSignature)
        
        # Metric for reflection quality (how general and effective the new instructions are)
        def reflection_quality_metric(example, prediction, trace=None):
            """Evaluate reflection quality based on instruction generality and effectiveness."""
            # Check if new instruction is general (doesn't contain specific content)
            new_instruction = getattr(prediction, 'new_instruction', '')
            
            # Penalty for task-specific terms
            specific_terms = ['heisenberg', 'uncertainty', 'principle', 'nash', 'equilibrium', 'dna', 'photosynthesis']
            specificity_penalty = sum(1 for term in specific_terms if term.lower() in new_instruction.lower()) * 0.2
            
            # Reward for structural improvements
            improvement_indicators = ['clear', 'concise', 'accurate', 'comprehensive', 'structured']
            improvement_score = sum(0.1 for indicator in improvement_indicators if indicator in new_instruction.lower())
            
            # Base score for having a valid instruction
            base_score = 0.5 if new_instruction and len(new_instruction) > 20 else 0.0
            
            final_score = min(1.0, base_score + improvement_score - specificity_penalty)
            return final_score
        
        # Choose optimizer
        if optimizer_type == "miprov2":
            optimizer = MIPROv2(
                metric=reflection_quality_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=5,
                verbose=False
            )
        else:  # bootstrap
            optimizer = BootstrapFewShot(
                metric=reflection_quality_metric,
                max_labeled_demos=8,
                max_rounds=3
            )
        
        logger.info(f"Optimizing reflection strategy using {optimizer_type} on {len(reflection_trainset)} examples")
        optimized_reflector = optimizer.compile(base_reflector, trainset=reflection_trainset)
        
        return GEPAReflection(optimized_reflector=optimized_reflector)
        
    except Exception as e:
        logger.warning(f"Failed to create optimized reflection strategy: {e}")
        return GEPAReflection()
