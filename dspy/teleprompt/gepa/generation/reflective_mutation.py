"""Reflective prompt mutation implementing core GEPA innovation."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Callable

import dspy
from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


class PromptMutator(ABC):
    """Protocol for reflective prompt mutation (core GEPA innovation)."""

    @abstractmethod
    def mutate_signature(self, current_signature: Signature, feedback: "FeedbackResult") -> Signature:
        """Mutate signature based on reflective feedback."""
        raise NotImplementedError


class PerformanceAnalysisSignature(dspy.Signature):
    """Analyze performance and identify improvement opportunities."""
    current_instruction: str = dspy.InputField(desc="Current instruction text")
    task_description: str = dspy.InputField(desc="Description of the task (input -> output fields)")
    performance_summary: str = dspy.InputField(desc="Summary of current performance metrics")
    failure_examples: str = dspy.InputField(desc="Examples of failures and their patterns")
    execution_traces: str = dspy.InputField(desc="Traces from system execution showing reasoning steps")
    
    analysis: str = dspy.OutputField(desc="Detailed analysis of why current instruction fails")
    improvement_strategy: str = dspy.OutputField(desc="Strategy for improving the instruction")


class InstructionImprovementSignature(dspy.Signature):
    """Improve instruction based on analysis and strategy."""
    current_instruction: str = dspy.InputField(desc="Current instruction text")
    analysis: str = dspy.InputField(desc="Analysis of current instruction's weaknesses")
    improvement_strategy: str = dspy.InputField(desc="Strategy for improvement")
    task_context: str = dspy.InputField(desc="Additional context about the task requirements")
    
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning for the new instruction")
    improved_instruction: str = dspy.OutputField(desc="The improved instruction text")
    key_changes: str = dspy.OutputField(desc="Summary of key changes made")


class ReflectiveMutation(PromptMutator):
    """Reflective prompt mutation using advanced DSPy Chain-of-Thought."""

    def __init__(self, prompt_model: Optional[Any] = None):
        self.prompt_model = prompt_model
        self.mutation_history = []
        
        # Advanced DSPy modules for reflection
        self.performance_analyzer = dspy.ChainOfThought(PerformanceAnalysisSignature)
        self.instruction_improver = dspy.ChainOfThought(InstructionImprovementSignature)

    def mutate_signature(self, current_signature: Signature, feedback: "FeedbackResult") -> Signature:
        """Mutate signature using advanced DSPy Chain-of-Thought reflection.

        This implements the core GEPA innovation: using execution traces and
        performance feedback to guide prompt evolution through natural language reflection.
        Uses DSPy's ChainOfThought for sophisticated reasoning about instruction improvement.
        """
        try:
            # Prepare inputs for DSPy reflection modules
            current_instruction = current_signature.instructions or "Answer the question."
            task_description, performance_summary, failure_examples, execution_traces = self._prepare_reflection_inputs(
                current_signature, feedback
            )

            # Step 1: Analyze performance using ChainOfThought
            with dspy.context(lm=self.prompt_model):
                analysis_result = self.performance_analyzer(
                    current_instruction=current_instruction,
                    task_description=task_description,
                    performance_summary=performance_summary,
                    failure_examples=failure_examples,
                    execution_traces=execution_traces
                )

                # Step 2: Improve instruction based on analysis using ChainOfThought  
                improvement_result = self.instruction_improver(
                    current_instruction=current_instruction,
                    analysis=analysis_result.analysis,
                    improvement_strategy=analysis_result.improvement_strategy,
                    task_context=task_description
                )

            # Create new signature with improved instruction
            new_signature = self._create_improved_signature(current_signature, improvement_result)
            
            # Log detailed mutation history
            self._log_mutation_history(current_signature, improvement_result, feedback, analysis_result)
            
            return new_signature

        except Exception as e:
            logger.warning(f"Advanced reflective mutation failed: {e}")
            # Fallback to simpler reflection or original signature
            return self._fallback_mutation(current_signature, feedback)

    def _prepare_reflection_inputs(self, signature: Signature, feedback: "FeedbackResult") -> tuple:
        """Prepare structured inputs for DSPy reflection modules."""
        # Extract field information
        input_fields = []
        output_fields = []
        
        for field_name, field_info in signature.fields.items():
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                field_type = field_info.json_schema_extra.get('__dspy_field_type')
                if field_type == 'input':
                    input_fields.append(field_name)
                elif field_type == 'output':
                    output_fields.append(field_name)
            else:
                # Fallback: assume last field is output
                field_names = list(signature.fields.keys())
                if field_name == field_names[-1]:
                    output_fields.append(field_name)
                else:
                    input_fields.append(field_name)
        
        # Task description
        task_description = f"Transform {', '.join(input_fields)} -> {', '.join(output_fields)}"
        
        # Performance summary with detailed metrics
        avg_score = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
        score_distribution = self._analyze_score_distribution(feedback.scores)
        performance_summary = f"""Average score: {avg_score:.3f} on {len(feedback.scores)} examples
Score distribution: {score_distribution}
Success rate: {sum(1 for s in feedback.scores if s > 0.5) / len(feedback.scores) * 100:.1f}%"""
        
        # Detailed failure analysis
        failures = [(diag, score) for diag, score in zip(feedback.diagnostics, feedback.scores) if score <= 0.5]
        failure_examples = self._create_failure_analysis(failures)
        
        # Execution traces analysis
        execution_traces = self._analyze_execution_traces(feedback.traces)
        
        return task_description, performance_summary, failure_examples, execution_traces
    
    def _analyze_score_distribution(self, scores: List[float]) -> str:
        """Analyze score distribution for better insights."""
        if not scores:
            return "No scores available"
        
        high_scores = sum(1 for s in scores if s > 0.8)
        medium_scores = sum(1 for s in scores if 0.5 < s <= 0.8)
        low_scores = sum(1 for s in scores if s <= 0.5)
        
        return f"High (>0.8): {high_scores}, Medium (0.5-0.8): {medium_scores}, Low (≤0.5): {low_scores}"
    
    def _create_failure_analysis(self, failures: List[tuple]) -> str:
        """Create detailed failure analysis."""
        if not failures:
            return "No significant failures observed"
        
        # Group similar failure patterns
        failure_patterns = {}
        for diag, score in failures[:5]:  # Analyze top 5 failures
            # Simple pattern detection - could be enhanced
            if "INCORRECT" in diag:
                failure_patterns.setdefault("Accuracy Issues", []).append(f"Score {score:.2f}: {diag}")
            elif "failed" in diag.lower():
                failure_patterns.setdefault("Execution Failures", []).append(f"Score {score:.2f}: {diag}")
            else:
                failure_patterns.setdefault("Other Issues", []).append(f"Score {score:.2f}: {diag}")
        
        analysis = "\n".join([
            f"{pattern}: {'; '.join(examples[:2])}" 
            for pattern, examples in failure_patterns.items()
        ])
        
        return analysis or "Unspecified failure patterns"
    
    def _analyze_execution_traces(self, traces: List[List]) -> str:
        """Analyze execution traces for reasoning patterns."""
        if not traces or not any(traces):
            return "No execution traces available"
        
        # Simple trace analysis - could be enhanced with more sophisticated pattern detection
        trace_analysis = f"""Execution patterns observed across {len(traces)} examples:
- Average trace length: {sum(len(t) for t in traces) / len(traces):.1f} steps
- Complex reasoning traces: {sum(1 for t in traces if len(t) > 3)}
- Simple reasoning traces: {sum(1 for t in traces if len(t) <= 3)}"""
        
        return trace_analysis
    
    def _create_improved_signature(self, current_signature: Signature, improvement_result) -> Signature:
        """Create new signature with improved instruction."""
        from dspy.signatures.signature import make_signature
        
        # Extract field names for signature creation
        input_fields = []
        output_fields = []
        
        for field_name, field_info in current_signature.fields.items():
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                field_type = field_info.json_schema_extra.get('__dspy_field_type')
                if field_type == 'input':
                    input_fields.append(field_name)
                elif field_type == 'output':
                    output_fields.append(field_name)
            else:
                # Fallback: assume last field is output
                field_names = list(current_signature.fields.keys())
                if field_name == field_names[-1]:
                    output_fields.append(field_name)
                else:
                    input_fields.append(field_name)
        
        # Create new signature with improved instruction
        field_signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
        new_signature = make_signature(field_signature, improvement_result.improved_instruction)
        
        return new_signature
    
    def _log_mutation_history(self, original_signature, improvement_result, feedback, analysis_result):
        """Log mutation history for debugging and analysis."""
        mutation_entry = {
            'original_instruction': original_signature.instructions,
            'improved_instruction': improvement_result.improved_instruction,
            'analysis': analysis_result.analysis,
            'strategy': analysis_result.improvement_strategy,
            'key_changes': improvement_result.key_changes,
            'performance_before': sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0,
            'examples_count': len(feedback.scores)
        }
        
        self.mutation_history.append(mutation_entry)
        
        # Keep only recent history
        if len(self.mutation_history) > 100:
            self.mutation_history = self.mutation_history[-50:]
        
        logger.info(f"Mutation applied: {improvement_result.key_changes}")
    
    def _fallback_mutation(self, current_signature: Signature, feedback: "FeedbackResult") -> Signature:
        """Fallback mutation strategy when advanced reflection fails."""
        try:
            # Simple prefix-based improvement
            current_instruction = current_signature.instructions or "Answer the question."
            avg_score = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
            
            if avg_score < 0.3:
                # Very poor performance - add basic guidance
                improved_instruction = f"Think carefully and {current_instruction.lower()}"
            elif avg_score < 0.6:
                # Moderate performance - add step-by-step guidance
                improved_instruction = f"Let's think step by step. {current_instruction}"
            else:
                # Good performance - add precision guidance
                improved_instruction = f"{current_instruction} Be precise and thorough."
            
            # Create new signature
            from dspy.signatures.signature import make_signature
            
            # Extract field structure
            field_names = list(current_signature.fields.keys())
            if len(field_names) >= 2:
                input_fields = field_names[:-1]
                output_fields = field_names[-1:]
                field_signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
                return make_signature(field_signature, improved_instruction)
            
            # Fallback to original signature if structure unclear
            return current_signature
            
        except Exception as e:
            logger.warning(f"Fallback mutation also failed: {e}")
            return current_signature