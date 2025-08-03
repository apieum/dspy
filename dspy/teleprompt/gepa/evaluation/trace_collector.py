"""Enhanced feedback collection implementing μf function from paper Section 3.2."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import dspy

from .feedback import EvaluationTrace, FeedbackResult, ModuleFeedback

logger = logging.getLogger(__name__)


class FeedbackCollector(ABC):
    """Protocol for enhanced feedback collection (μf function)."""

    @abstractmethod
    def collect_feedback(self, program: dspy.Module, examples: List[dspy.Example], metric: Callable) -> FeedbackResult:
        """Collect enhanced feedback with traces and diagnostics."""
        raise NotImplementedError


class EnhancedTraceCollector(FeedbackCollector):
    """Enhanced feedback collector implementing Enhanced Feedback Function μf from paper Section 3.2."""

    def __init__(self, collect_module_feedback=True, collect_evaluation_traces=True):
        """Initialize enhanced feedback collector.
        
        Args:
            collect_module_feedback: Whether to collect per-module feedback
            collect_evaluation_traces: Whether to collect rich evaluation traces
        """
        self.collect_module_feedback = collect_module_feedback
        self.collect_evaluation_traces = collect_evaluation_traces
        self.domain_handlers = {}

    def collect_feedback(self, program: dspy.Module, examples: List[dspy.Example], metric: Callable) -> FeedbackResult:
        """Collect enhanced feedback with DSPy traces and diagnostics.

        Implements Enhanced Feedback Function μf with:
        - traces: DSPy execution traces for each example
        - diagnostics: Human-readable diagnostic messages
        - scores: Scalar scores from metric evaluation
        - evaluation_traces: Rich evaluation traces with compilation errors, execution steps
        - module_feedback: Module-level feedback for multi-hop systems
        - feedback_text: Textual diagnostic feedback
        """
        if not examples:
            return FeedbackResult(
                traces=[], diagnostics=[], scores=[],
                evaluation_traces=[], module_feedback=[], feedback_text=[]
            )

        traces = []
        diagnostics = []
        scores = []
        evaluation_traces = []
        module_feedback = []
        feedback_text = []

        for i, example in enumerate(examples):
            try:
                # Collect trace during execution
                trace = []
                compilation_errors = []
                execution_steps = []
                module_outputs = {}
                reasoning_chains = []
                error_messages = []
                
                start_time = time.time()
                
                with dspy.context(trace=trace):
                    try:
                        prediction = program(**example.inputs())
                        execution_steps.append(f"Successfully executed program with inputs: {list(example.inputs().keys())}")
                    except Exception as exec_error:
                        error_messages.append(f"Execution error: {str(exec_error)}")
                        prediction = type('EmptyPrediction', (), {})()
                
                execution_time = time.time() - start_time

                # Extract trace from context
                execution_trace = trace
                traces.append(execution_trace)

                # Collect module outputs and reasoning chains
                for j, step in enumerate(execution_trace):
                    if hasattr(step, 'outputs'):
                        module_outputs[j] = step.outputs
                    if hasattr(step, 'reasoning'):
                        reasoning_chains.append(step.reasoning)

                # Compute score
                try:
                    score = metric(example, prediction, execution_trace)
                except TypeError:
                    # Fallback for metrics that don't accept trace parameter
                    score = metric(example, prediction)
                scores.append(float(score))

                # Generate diagnostic message
                diagnostic = self._generate_diagnostic(example, prediction, score, execution_trace)
                diagnostics.append(diagnostic)

                # Enhanced Feedback Function μf: Create rich evaluation trace
                if self.collect_evaluation_traces:
                    eval_trace = EvaluationTrace(
                        execution_steps=execution_steps,
                        compilation_errors=compilation_errors,
                        intermediate_outputs=[getattr(step, 'outputs', None) for step in execution_trace],
                        module_outputs=module_outputs,
                        reasoning_chains=reasoning_chains,
                        tool_calls=[],  # Could be enhanced to track actual tool calls
                        error_messages=error_messages,
                        performance_metrics={
                            "score": float(score),
                            "execution_time": execution_time,
                            "trace_length": len(execution_trace)
                        }
                    )
                    evaluation_traces.append(eval_trace)

                # Enhanced Feedback Function μf: Create module-level feedback
                if self.collect_module_feedback:
                    for j, step in enumerate(execution_trace):
                        module_fb = ModuleFeedback(
                            module_id=j,
                            module_name=getattr(step, '__class__', type(step)).__name__,
                            input_data=getattr(step, 'inputs', {}),
                            output_data=getattr(step, 'outputs', {}),
                            execution_time=execution_time / max(1, len(execution_trace)),
                            success=len(error_messages) == 0,
                            error_message=error_messages[0] if error_messages else None,
                            intermediate_reasoning=reasoning_chains,
                            confidence_score=float(score)
                        )
                        module_feedback.append(module_fb)

                # Enhanced Feedback Function μf: Generate textual feedback
                feedback_text.append(self._generate_textual_feedback(example, prediction, score, eval_trace if self.collect_evaluation_traces else None))

            except Exception as e:
                logger.warning(f"Failed to collect feedback for example {i}: {e}")
                traces.append([])
                scores.append(0.0)
                diagnostics.append(f"Execution failed: {str(e)}")
                
                # Add empty enhanced feedback for failed examples
                if self.collect_evaluation_traces:
                    evaluation_traces.append(EvaluationTrace(
                        execution_steps=[f"Failed: {str(e)}"],
                        compilation_errors=[str(e)],
                        intermediate_outputs=[],
                        module_outputs={},
                        reasoning_chains=[],
                        tool_calls=[],
                        error_messages=[str(e)],
                        performance_metrics={"score": 0.0, "execution_time": 0.0, "trace_length": 0}
                    ))
                
                if self.collect_module_feedback:
                    module_feedback.append(ModuleFeedback(
                        module_id=0,
                        module_name="FailedExecution",
                        input_data=example.inputs(),
                        output_data={},
                        execution_time=0.0,
                        success=False,
                        error_message=str(e),
                        intermediate_reasoning=[],
                        confidence_score=0.0
                    ))
                
                feedback_text.append(f"Execution failed: {str(e)}")

        return FeedbackResult(
            traces=traces, 
            diagnostics=diagnostics, 
            scores=scores,
            evaluation_traces=evaluation_traces,
            module_feedback=module_feedback,
            feedback_text=feedback_text
        )

    def _generate_textual_feedback(self, example: dspy.Example, prediction: Any, score: float, eval_trace: Optional[EvaluationTrace]) -> str:
        """Generate rich textual feedback using domain-specific handlers."""
        # Detect domain
        domain = self._detect_domain(example)
        
        # Use domain-specific handler if available
        if domain in self.domain_handlers:
            try:
                domain_feedback = self.domain_handlers[domain](example, prediction, eval_trace)
                return f"Domain ({domain}): {domain_feedback}"
            except Exception as e:
                logger.warning(f"Domain handler {domain} failed: {e}")
        
        # Generate generic feedback
        status = "SUCCESS" if score > 0.5 else "FAILURE"
        feedback_parts = [f"Status: {status} (Score: {score:.2f})"]
        
        if eval_trace:
            if eval_trace.error_messages:
                feedback_parts.append(f"Errors: {'; '.join(eval_trace.error_messages[:3])}")
            if eval_trace.reasoning_chains:
                feedback_parts.append(f"Reasoning steps: {len(eval_trace.reasoning_chains)}")
            
            exec_time = eval_trace.performance_metrics.get("execution_time", 0.0)
            feedback_parts.append(f"Execution time: {exec_time:.3f}s")
        
        return " | ".join(feedback_parts)

    def register_domain_handler(self, domain: str, handler: Callable):
        """Register a domain-specific feedback handler."""
        self.domain_handlers[domain] = handler

    def _detect_domain(self, example: dspy.Example) -> str:
        """Detect the domain of an example for specialized feedback."""
        # Simple heuristic-based domain detection
        inputs = example.inputs()
        
        # Check for code-related keywords
        text_content = " ".join(str(v).lower() for v in inputs.values())
        if any(keyword in text_content for keyword in ['code', 'function', 'class', 'python', 'javascript']):
            return 'code'
        
        # Check for math-related keywords
        if any(keyword in text_content for keyword in ['calculate', 'solve', 'equation', 'math', 'number']):
            return 'math'
        
        # Check for reasoning-related keywords
        if any(keyword in text_content for keyword in ['because', 'therefore', 'reasoning', 'explain', 'why']):
            return 'reasoning'
        
        return 'general'

    def _generate_diagnostic(self, example: dspy.Example, prediction: Any, score: float, trace: List) -> str:
        """Generate human-readable diagnostic message."""
        # Basic diagnostic - can be enhanced with more sophisticated analysis
        status = "CORRECT" if score > 0.5 else "INCORRECT"

        # Extract key information from prediction
        pred_summary = str(prediction)[:100] + "..." if len(str(prediction)) > 100 else str(prediction)

        # Count reasoning steps from trace
        reasoning_steps = len([step for step in trace if hasattr(step, 'reasoning')])

        diagnostic = f"[{status}] Score: {score:.2f}, Steps: {reasoning_steps}, Prediction: {pred_summary}"

        # Add specific failure analysis for low scores
        if score <= 0.5:
            expected = getattr(example, 'answer', 'N/A')
            diagnostic += f" | Expected: {expected}"

        return diagnostic