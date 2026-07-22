"""Feedback provider for GEPA optimization."""

import logging
import inspect
from typing import Any, Callable, List, Optional
from ..evaluation import Assessor
import dspy

logger = logging.getLogger(__name__)


class FeedbackProvider:
    """Encapsulates metric (μ) and enhanced feedback function (μf) for GEPA.

    This class provides a clean interface for evaluation and diagnostic feedback,
    supporting μf-compliant metrics that return detailed evaluation traces
    for intelligent reflection.

    The metric returns (float, str): Score and rich feedback text (μf-compliant).
    """

    def __init__(
        self,
        assessor: Optional[Assessor] = None,
        feedback_function: Optional[Callable] = None,
        metric: Optional[Callable] = None,
    ):
        """Initialize feedback provider.

        Args:
            assessor: Evaluation function μ that returns (float, str): Score and rich diagnostic text (μf-compliant)
            feedback_function: Optional enhanced feedback function μf for additional diagnostics
        """
        # ``metric`` was the original Darwin name. Keep it as an explicit
        # alias while using ``assessor`` in the component API.
        if assessor is None:
            assessor = metric
        if assessor is None:
            raise ValueError("FeedbackProvider requires an assessor function")

        self.assessor = assessor
        self.feedback_function = feedback_function

    @staticmethod
    def _call_feedback(function: Callable, *args):
        """Call μf functions using the richest signature they declare."""
        try:
            parameters = inspect.signature(function).parameters.values()
            positional = [
                parameter
                for parameter in parameters
                if parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(
                parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
            )
            count = len(args) if has_varargs else len(positional)
        except (TypeError, ValueError):
            count = len(args)
        return function(*list(args)[:count])

    @classmethod
    def _call_assessor(cls, function: Callable, example, prediction, trace, pred_name, pred_trace):
        """Support both normal DSPy metrics and GEPA's five-argument API."""
        return cls._call_feedback(function, example, prediction, trace, pred_name, pred_trace)

    def evaluate(self, example: dspy.Example, prediction, trace: Optional[List] = None,
                module_idx: Optional[int] = None, pred_name: Optional[str] = None,
                pred_trace: Optional[Any] = None) -> tuple[float, str]:
        """Evaluate example and provide feedback, now capturing rich µf output.

        Args:
            example: Training example
            prediction: Model prediction
            trace: Execution trace (optional)
            module_idx: Target module index for module-specific feedback (optional)

        Returns:
            Tuple of (score, diagnostic_text)
        """
        if pred_trace is None and isinstance(trace, list) and module_idx is not None:
            if 0 <= module_idx < len(trace):
                pred_trace = trace[module_idx]
        metric_result = self._call_assessor(
            self.assessor, example, prediction, trace,
            pred_name if pred_name is not None else (
                str(module_idx) if module_idx is not None else None
            ), pred_trace,
        )
        if isinstance(metric_result, tuple) and len(metric_result) == 2:
            metric_score, metric_feedback = metric_result
            score = float(metric_score)
            feedback_text = str(metric_feedback)
        else:
            score = float(metric_result)
            feedback_text = str(metric_result)

        status = "SUCCESS" if score > 0.5 else "FAILURE"

        # Enhanced: Use Metric's rich feedback capabilities
        diagnostic_parts = [f"Score: {score:.2f} ({status})"]

        if feedback_text:
            diagnostic_parts.append(f"Feedback: {feedback_text}")

        # Include suggestions if available
        if hasattr(metric_result, 'suggestions') and metric_result.suggestions:
            suggestions_text = "; ".join(metric_result.suggestions[:3])  # Limit to first 3
            diagnostic_parts.append(f"Suggestions: {suggestions_text}")

        # Include error analysis if available
        if hasattr(metric_result, 'error_analysis') and metric_result.error_analysis:
            error_type = metric_result.error_analysis.get('error_type', 'unknown')
            diagnostic_parts.append(f"Error Type: {error_type}")

        diagnostic = " | ".join(diagnostic_parts)

        # Get additional diagnostic feedback from enhanced feedback function
        if self.feedback_function:
            try:
                # Enhanced feedback function (μf) provides additional rich diagnostics
                feedback_result = self._call_feedback(
                    self.feedback_function, example, prediction, trace,
                    pred_name if pred_name is not None else (
                        str(module_idx) if module_idx is not None else None
                    ), pred_trace,
                )

                if isinstance(feedback_result, tuple) and len(feedback_result) == 2:
                    # μf can override score but we append diagnostic_text
                    enhanced_score, additional_diagnostic = feedback_result
                    score = float(enhanced_score)
                    diagnostic = f"Score: {score:.2f} ({status}) | {additional_diagnostic}"
                elif isinstance(feedback_result, dict):
                    # Rich feedback dictionary
                    additional_diagnostic = self._format_rich_feedback(feedback_result, score)
                    diagnostic += f" | {additional_diagnostic}"
                else:
                    # Simple diagnostic text - append to existing
                    diagnostic += f" | {str(feedback_result)}"
            except Exception as e:
                logger.warning(f"Enhanced feedback function failed: {e}")
                diagnostic += f" | Enhanced feedback failed: {str(e)}"

        return score, diagnostic

    def _format_rich_feedback(self, feedback_dict: dict, score: float) -> str:
        """Format rich feedback dictionary from enhanced feedback function."""
        parts = [f"Score: {score:.2f}"]

        # Common rich feedback fields
        if 'error_type' in feedback_dict:
            parts.append(f"Error Type: {feedback_dict['error_type']}")
        if 'error_location' in feedback_dict:
            parts.append(f"Error Location: {feedback_dict['error_location']}")
        if 'suggestion' in feedback_dict:
            parts.append(f"Suggestion: {feedback_dict['suggestion']}")
        if 'evaluation_traces' in feedback_dict:
            parts.append(f"Eval Traces: {feedback_dict['evaluation_traces']}")
        if 'module_feedback' in feedback_dict:
            parts.append(f"Module Feedback: {feedback_dict['module_feedback']}")

        return " | ".join(parts)
