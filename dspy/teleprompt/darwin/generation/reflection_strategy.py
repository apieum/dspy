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
    """GEPA's reflection and prompt update based on paper's Appendix B.
    
    This signature implements the exact meta-prompt from the paper, using
    DSPy's structured format to ensure consistent reflection.
    """
    current_instruction: str = dspy.InputField(
        desc="The current instruction provided to the assistant"
    )
    formatted_examples: str = dspy.InputField(
        desc="Examples showing inputs, assistant outputs, and feedback on how responses could be better"
    )
    
    task_analysis: str = dspy.OutputField(
        desc="Detailed analysis of the task format, requirements, and patterns observed in the examples"
    )
    improvement_strategy: str = dspy.OutputField(
        desc="Strategy for improving the instruction based on successes, failures, and domain-specific patterns"
    )
    new_instruction: str = dspy.OutputField(
        desc="The improved instruction that addresses identified issues and incorporates learned patterns"
    )


class GEPAReflection(ReflectionStrategy):
    """Paper-compliant GEPA reflection using DSPy ChainOfThought.
    
    This is the default reflection strategy implementing the exact
    approach described in the GEPA paper with natural language reflection.
    """
    
    def __init__(self):
        # DSPy ChainOfThought for reflection with paper's approach
        # ChainOfThought automatically adds 'reasoning' field for step-by-step thinking
        self.reflector = dspy.ChainOfThought(GEPAReflectionSignature)
    
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
            if hasattr(reflection_result, 'task_analysis'):
                logger.debug(f"Task analysis: {reflection_result.task_analysis[:200]}...")
            if hasattr(reflection_result, 'improvement_strategy'):
                logger.debug(f"Improvement strategy: {reflection_result.improvement_strategy[:200]}...")
            
            # Return the improved instruction
            return reflection_result.new_instruction
            
        except Exception as e:
            logger.warning(f"GEPA reflection failed: {e}, returning original instruction")
            return current_instruction


class SimpleReflection(ReflectionStrategy):
    """Lightweight reflection strategy without structured reasoning.
    
    This strategy uses a simple prompt for faster, less expensive reflection.
    Useful for quick iterations or resource-constrained scenarios.
    """
    
    def reflect(self, 
                current_instruction: str,
                formatted_examples: str,
                prompt_model: Optional[Any] = None) -> str:
        """Simple reflection using basic prompting."""
        try:
            prompt = f"""Given this instruction:
{current_instruction}

And these examples with feedback:
{formatted_examples}

Write an improved instruction that addresses the feedback. Be concise.

Improved instruction:"""

            # Use simple Predict for lightweight reflection
            predictor = dspy.Predict("prompt -> improved_instruction")
            
            with dspy.context(lm=prompt_model) if prompt_model else dspy.context():
                result = predictor(prompt=prompt)
            
            return result.improved_instruction
            
        except Exception as e:
            logger.warning(f"Simple reflection failed: {e}, returning original instruction")
            return current_instruction


class PrefixReflection(ReflectionStrategy):
    """Basic prefix-based reflection without LLM calls.
    
    This strategy adds simple prefixes based on performance,
    useful as a baseline or when LLM calls are expensive.
    """
    
    def reflect(self, 
                current_instruction: str,
                formatted_examples: str,
                prompt_model: Optional[Any] = None) -> str:
        """Add performance-based prefixes to instruction."""
        # Analyze feedback to determine performance level
        success_count = formatted_examples.count("Good response")
        failure_count = formatted_examples.count("Needs improvement")
        
        if failure_count > success_count * 2:
            # Many failures - add careful thinking guidance
            return f"Think carefully and systematically. {current_instruction}"
        elif failure_count > success_count:
            # Moderate performance - add step-by-step guidance
            return f"Let's approach this step by step. {current_instruction}"
        else:
            # Good performance - add precision guidance
            return f"{current_instruction} Be precise and thorough in your response."