"""Feedback provider for evaluation with rich diagnostics.

Provides a clean interface for evaluation and diagnostic feedback,
supporting metrics that return detailed evaluation traces for
intelligent optimization.
"""

import logging
import inspect
from typing import Any, Callable, List, Optional, Dict
import dspy

from .metrics import Assessor


logger = logging.getLogger(__name__)


class FeedbackProvider:
    """Encapsulates metric and enhanced feedback function for optimization.

    Provides a flexible interface for evaluation with rich diagnostic feedback.
    Supports both simple metrics (score only) and enhanced metrics (score + feedback).

    The assessor can return:
    - float: Simple score
    - (float, str): Score and feedback text
    - Metric object: With value, feedback, suggestions, etc.

    Attributes:
        assessor: Evaluation function that returns score and optionally feedback
        feedback_function: Optional enhanced feedback function for additional diagnostics
        failure_score: Score to return on evaluation failure
    """

    def __init__(
        self,
        assessor: Optional[Assessor] = None,
        feedback_function: Optional[Callable] = None,
        metric: Optional[Callable] = None,
        failure_score: float = 0.0,
    ):
        """Initialize feedback provider.

        Args:
            assessor: Evaluation function
            feedback_function: Optional enhanced feedback function for diagnostics
            metric: Alias for assessor (backward compatibility)
            failure_score: Score to return on evaluation failure
        """
        if assessor is None:
            assessor = metric

        if assessor is None:
            raise ValueError("FeedbackProvider requires an assessor function")

        self.assessor = assessor
        self.feedback_function = feedback_function
        self.failure_score = float(failure_score)

    @staticmethod
    def _call_feedback(function: Callable, *args):
        """Call functions using the richest signature they declare.

        Supports variable signatures by inspecting function parameters
        and passing only the arguments it can accept.

        Args:
            function: Function to call
            *args: Arguments to pass

        Returns:
            Function result
        """
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
    def _call_assessor(
        cls,
        function: Callable,
        example,
        prediction,
        trace,
        context_id,
        context_trace
    ):
        """Call assessor with flexible signature support.

        Supports assessors with different signatures:
        - (example, prediction)
        - (example, prediction, trace)
        - (example, prediction, trace, context_id, context_trace)

        Args:
            function: Assessor function
            example: Training example
            prediction: Model prediction
            trace: Execution trace
            context_id: Optional context identifier (module name, task id, etc.)
            context_trace: Optional context-specific trace

        Returns:
            Assessor result
        """
        return cls._call_feedback(
            function,
            example,
            prediction,
            trace,
            context_id,
            context_trace
        )

    def evaluate(
        self,
        example: dspy.Example,
        prediction,
        trace: Optional[List] = None,
        context_id: Optional[Any] = None,
        context_trace: Optional[Any] = None
    ) -> tuple[float, str]:
        """Evaluate example and provide feedback.

        Args:
            example: Training example
            prediction: Model prediction
            trace: Optional execution trace
            context_id: Optional context identifier (module index, task id, etc.)
            context_trace: Optional context-specific trace

        Returns:
            Tuple of (score, diagnostic_text)
        """
        score, diagnostic, _ = self.evaluate_rich(
            example,
            prediction,
            trace,
            context_id,
            context_trace
        )
        return score, diagnostic

    def evaluate_rich(
        self,
        example: dspy.Example,
        prediction,
        trace: Optional[List] = None,
        context_id: Optional[Any] = None,
        context_trace: Optional[Any] = None
    ) -> tuple[float, str, Any]:
        """Evaluate example with rich feedback.

        Args:
            example: Training example
            prediction: Model prediction
            trace: Optional execution trace
            context_id: Optional context identifier
            context_trace: Optional context-specific trace

        Returns:
            Tuple of (score, diagnostic_text, side_info)
        """
        metric_result = self._call_assessor(
            self.assessor,
            example,
            prediction,
            trace,
            context_id,
            context_trace
        )

        score, feedback_text = self._extract_score_and_feedback(metric_result)

        status = "SUCCESS" if score > 0.5 else "FAILURE"
        diagnostic_parts = [f"Score: {score:.2f} ({status})"]

        if feedback_text:
            diagnostic_parts.append(f"Feedback: {feedback_text}")

        diagnostic_parts.extend(
            self._extract_metric_diagnostics(metric_result)
        )

        diagnostic = " | ".join(diagnostic_parts)

        if self.feedback_function:
            diagnostic = self._apply_enhanced_feedback(
                diagnostic,
                example,
                prediction,
                trace,
                context_id,
                context_trace,
                score,
                status
            )

        side_info = getattr(metric_result, "side_info", None)

        return score, diagnostic, side_info

    def _extract_score_and_feedback(
        self,
        metric_result: Any
    ) -> tuple[float, str]:
        """Extract score and feedback from metric result.

        Args:
            metric_result: Result from assessor

        Returns:
            Tuple of (score, feedback_text)
        """
        if isinstance(metric_result, tuple) and len(metric_result) == 2:
            metric_score, metric_feedback = metric_result
            score = float(metric_score)
            feedback_text = str(metric_feedback)
        else:
            score = float(metric_result)
            feedback_text = str(metric_result)

        return score, feedback_text

    def _extract_metric_diagnostics(self, metric_result: Any) -> List[str]:
        """Extract diagnostic information from metric result.

        Args:
            metric_result: Result from assessor

        Returns:
            List of diagnostic strings
        """
        diagnostics = []

        if hasattr(metric_result, 'suggestions') and metric_result.suggestions:
            suggestions_text = "; ".join(metric_result.suggestions[:3])
            diagnostics.append(f"Suggestions: {suggestions_text}")

        if hasattr(metric_result, 'errors') and metric_result.errors:
            error_type = metric_result.errors.get('error_type', 'unknown')
            diagnostics.append(f"Error Type: {error_type}")

        return diagnostics

    def _apply_enhanced_feedback(
        self,
        diagnostic: str,
        example: dspy.Example,
        prediction,
        trace: Optional[List],
        context_id: Optional[Any],
        context_trace: Optional[Any],
        score: float,
        status: str
    ) -> str:
        """Apply enhanced feedback function if available.

        Args:
            diagnostic: Current diagnostic string
            example: Training example
            prediction: Model prediction
            trace: Execution trace
            context_id: Context identifier
            context_trace: Context-specific trace
            score: Evaluation score
            status: Status string

        Returns:
            Enhanced diagnostic string
        """
        try:
            feedback_result = self._call_feedback(
                self.feedback_function,
                example,
                prediction,
                trace,
                context_id,
                context_trace
            )

            if isinstance(feedback_result, tuple) and len(feedback_result) == 2:
                enhanced_score, additional_diagnostic = feedback_result
                score = float(enhanced_score)
                diagnostic = f"Score: {score:.2f} ({status}) | {additional_diagnostic}"
            elif isinstance(feedback_result, dict):
                additional_diagnostic = self._format_rich_feedback(
                    feedback_result,
                    score
                )
                diagnostic += f" | {additional_diagnostic}"
            else:
                diagnostic += f" | {str(feedback_result)}"

        except Exception as e:
            logger.warning(f"Enhanced feedback function failed: {e}")
            diagnostic += f" | Enhanced feedback failed: {str(e)}"

        return diagnostic

    def _format_rich_feedback(self, feedback_dict: Dict, score: float) -> str:
        """Format rich feedback dictionary from enhanced feedback function.

        Args:
            feedback_dict: Dictionary with feedback information
            score: Evaluation score

        Returns:
            Formatted feedback string
        """
        parts = [f"Score: {score:.2f}"]

        common_fields = [
            'error_type',
            'error_location',
            'suggestion',
            'evaluation_traces',
            'module_feedback',
            'context_feedback'
        ]

        for field in common_fields:
            if field in feedback_dict:
                field_name = field.replace('_', ' ').title()
                parts.append(f"{field_name}: {feedback_dict[field]}")

        return " | ".join(parts)
