"""Prompt mutation strategies using DSPy's native systems."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import dspy
from dspy import Module
from dspy.signatures.signature import make_signature

from .reflection_strategy import ReflectionStrategy, GEPAReflection
from .instruction_updater import InstructionUpdater
from .error_handling import safe_deepcopy
from .dspy_utils import get_valid_predictor, get_predictor_instruction
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
        """Initialize reflective mutator with pre-optimized reflection strategy.

        Args:
            reflection_strategy: Strategy for generating improved instructions (defaults to optimized GEPA reflection)
            reflection_lm: Language model for reflection (passed to strategy)
        """
        # Use pre-optimized reflection strategy by default
        self.reflection_strategy = reflection_strategy or GEPAReflection()
        self.reflection_lm = reflection_lm
        self.mutation_count = 0

    def mutate(self, module: Module, feedback: FeedbackResult, target_module_idx: int = 0, verbose: bool = False) -> Module:
        """Mutate module using reflective prompt improvement with DSPy native systems."""
        # Use safe deepcopy for mutation
        mutated_module = safe_deepcopy(module, "Reflective mutation deepcopy failed")
        
        # Get valid target predictor
        target_predictor = get_valid_predictor(mutated_module, target_module_idx)
        if not target_predictor:
            logger.warning("No valid predictor to mutate")
            return mutated_module

        # Get current instruction from DSPy signature
        current_instruction = get_predictor_instruction(target_predictor)

        # Use enhanced abstract feedback formatting (prevents task-specific overfitting)
        formatted_examples = self._format_enhanced_feedback(feedback, target_module_idx)

        # Use reflection strategy to generate improved instruction
        improved_instruction = self.reflection_strategy.reflect(
            current_instruction=current_instruction,
            formatted_examples=formatted_examples,
            prompt_model=self.reflection_lm
        )

        # Display instruction evolution if verbose mode is enabled
        if verbose:
            self._display_instruction_evolution(current_instruction, improved_instruction, target_module_idx, feedback, formatted_examples)

        # Apply improved instruction using shared utility
        InstructionUpdater.update_instruction(target_predictor, improved_instruction)

        # Track mutation
        self.mutation_count += 1
        if verbose:
            print(f"Applied reflective mutation #{self.mutation_count} to predictor {target_module_idx}\n")
        else:
            logger.debug(f"Applied reflective mutation #{self.mutation_count}: {current_instruction[:50]}... -> {improved_instruction[:50]}...")

        return mutated_module

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
                    # This section processes the execution trace of a specific predictor for a given example.
                    # A DSPy trace for a predictor call is a tuple: (predictor_object, inputs_dict, outputs_dict).
                    # We unpack this tuple to access the inputs and outputs.
                    # `feedback.traces[i]` gets the trace for the i-th example in the batch.
                    # `[target_module_idx]` selects the specific predictor's trace we want to analyze.
                    predictor, inputs, outputs = feedback.traces[i][target_module_idx]

                    # The following lines format the `inputs` and `outputs` dictionaries into
                    # a concise, human-readable string for the reflection model.

                    # It iterates through the input dictionary. For each key-value pair, it creates a "key: value" string.
                    # If a value is too long (over 30 characters), it's truncated to keep the context manageable.
                    # All these strings are then joined together.
                    input_str = ", ".join([f"{k}: {str(v)[:30]}..." if len(str(v)) > 30 else f"{k}: {v}"
                                         for k, v in inputs.items()])

                    # Enhanced output formatting that preserves reasoning information
                    output_parts = []
                    reasoning_text = None

                    for k, v in outputs.items():
                        if k.lower() in ['reasoning', 'rationale', 'thought']:
                            # Preserve full reasoning for reflection (up to 200 chars)
                            reasoning_text = str(v)[:200] + ("..." if len(str(v)) > 200 else "")
                            output_parts.append(f"{k}: {reasoning_text}")
                        else:
                            # Standard truncation for other fields
                            truncated = str(v)[:30] + "..." if len(str(v)) > 30 else str(v)
                            output_parts.append(f"{k}: {truncated}")

                    output_str = ", ".join(output_parts)

                    # Enhanced trace info that highlights reasoning when available
                    if reasoning_text:
                        trace_info = f"Input: {input_str} → Output: {output_str}\nReasoning: {reasoning_text}"
                    else:
                        trace_info = f"Input: {input_str} → Output: {output_str}"

            example_text = f"""Example {i+1}:
Score: {score:.2f}
Feedback: {diagnostic}
Execution: {trace_info}"""
            formatted_parts.append(example_text)

        return "\n\n".join(formatted_parts)

    def _format_enhanced_feedback(self, feedback: FeedbackResult, target_module_idx: int) -> str:
        """Enhanced feedback formatting optimized for generalization and quality.
        
        This method provides structured performance insights without content-specific details,
        enabling effective instruction improvement while preventing overfitting.
        """
        if not feedback.scores or not feedback.diagnostics:
            return "No performance data available for analysis."

        formatted_parts = []
        total_score = sum(feedback.scores) / len(feedback.scores)
        
        # Performance summary
        summary_level = "HIGH" if total_score >= 0.8 else "MODERATE" if total_score >= 0.5 else "LOW"
        formatted_parts.append(f"Performance Summary: {summary_level} (μ={total_score:.2f}, n={len(feedback.scores)})")
        
        for i, (score, diagnostic) in enumerate(zip(feedback.scores, feedback.diagnostics)):
            # Performance classification
            status = "SUCCESS" if score >= 0.8 else "PARTIAL" if score >= 0.4 else "FAILURE"
            
            # Structural analysis without content exposure
            response_analysis = "Structure unknown"
            if (feedback.traces and i < len(feedback.traces) and
                feedback.traces[i] and target_module_idx < len(feedback.traces[i])):
                    predictor, inputs, outputs = feedback.traces[i][target_module_idx]
                    
                    # Extract structural characteristics
                    input_schema = list(inputs.keys())
                    output_schema = list(outputs.keys())
                    
                    # Analyze response quality indicators
                    primary_output = ""
                    if outputs:
                        primary_key = 'answer' if 'answer' in outputs else list(outputs.keys())[0]
                        primary_output = str(outputs.get(primary_key, ""))
                    
                    word_count = len(primary_output.split()) if primary_output else 0
                    has_reasoning = any(k.lower() in ['reasoning', 'rationale', 'thought', 'explanation'] for k in outputs.keys())
                    
                    # Quality indicators
                    completeness = "complete" if word_count > 5 else "brief" if word_count > 0 else "empty"
                    reasoning_present = "with reasoning" if has_reasoning else "direct answer"
                    
                    response_analysis = f"Schema: {input_schema} → {output_schema}, Response: {completeness} ({word_count}w), Mode: {reasoning_present}"

            # Quality assessment without revealing content
            case_feedback = f"""Case {i+1}: {status} (σ={score:.2f})
  Issue: {diagnostic}
  Analysis: {response_analysis}"""
            formatted_parts.append(case_feedback)

        return "\n".join(formatted_parts)

    def _display_instruction_evolution(self, old_instruction: str, new_instruction: str, target_module_idx: int, feedback: FeedbackResult, formatted_examples: str = None) -> None:
        """Display instruction evolution in verbose mode."""

        # Calculate average score for context
        avg_score = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0

        print(f"\n{'='*80}")
        print(f"INSTRUCTION EVOLUTION - Predictor {target_module_idx}")
        print(f"{'='*80}")
        print(f"Feedback Score: {avg_score:.3f} (n={len(feedback.scores)})")
        print(f"Mutation: #{self.mutation_count + 1}")
        print()

        # Display reflection input if available
        if formatted_examples:
            print("REFLECTION INPUT (Feedback for Analysis):")
            print("-" * 40)
            # Show first few lines of formatted examples to understand what reflection received
            preview_lines = formatted_examples.split('\n')[:10]  # First 10 lines
            for line in preview_lines:
                print(f"  {line}")
            if len(formatted_examples.split('\n')) > 10:
                print("  [... additional feedback examples ...]")
            print()

        # Display old instruction
        print("CURRENT INSTRUCTION:")
        print("-" * 40)
        print(f"  {old_instruction}")
        print()

        # Display new instruction
        print("EVOLVED INSTRUCTION:")
        print("-" * 40)
        print(f"  {new_instruction}")
        print()

        # Show key changes if instructions are different
        if old_instruction.strip() != new_instruction.strip():
            print("DETECTED CHANGES:")
            print("-" * 40)

            # Simple diff-like analysis
            old_words = set(old_instruction.lower().split())
            new_words = set(new_instruction.lower().split())

            added_words = new_words - old_words
            removed_words = old_words - new_words

            if added_words:
                print(f"  Added terms: {', '.join(sorted(added_words))}")
            if removed_words:
                print(f"  Removed terms: {', '.join(sorted(removed_words))}")

            if not added_words and not removed_words:
                print("  Structural or semantic modification detected")
        else:
            print("No textual changes detected")

        print("=" * 80)
        print()






