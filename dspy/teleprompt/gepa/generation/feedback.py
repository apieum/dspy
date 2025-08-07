"""Feedback provider for GEPA optimization."""

import logging
from typing import Any, Callable, List, Optional

import dspy

logger = logging.getLogger(__name__)


class FeedbackProvider:
    """Encapsulates metric (μ) and enhanced feedback function (μf) for GEPA.
    
    This class provides a clean interface for evaluation and diagnostic feedback,
    allowing easy experimentation with different feedback strategies.
    """
    
    def __init__(self, metric: Callable, feedback_function: Optional[Callable] = None):
        """Initialize feedback provider.
        
        Args:
            metric: Evaluation function μ that returns score
            feedback_function: Optional enhanced feedback function μf for rich diagnostics
        """
        if metric is None:
            raise ValueError("FeedbackProvider requires a metric function")
        
        self.metric = metric
        self.feedback_function = feedback_function
    
    def evaluate(self, example: dspy.Example, prediction, trace: Optional[List] = None, 
                module_idx: Optional[int] = None) -> tuple[float, str]:
        """Evaluate example and provide feedback.
        
        Args:
            example: Training example
            prediction: Model prediction
            trace: Execution trace (optional)
            module_idx: Target module index for module-specific feedback (optional)
            
        Returns:
            Tuple of (score, diagnostic_text)
        """
        # Get score using metric (μ)
        try:
            score = float(self.metric(example, prediction, trace))
        except TypeError:
            # Metric might not accept trace parameter
            score = float(self.metric(example, prediction))
        
        # Get diagnostic feedback
        if self.feedback_function:
            try:
                # Enhanced feedback function (μf) provides rich diagnostics
                feedback_result = self.feedback_function(example, prediction, trace, module_idx)
                
                if isinstance(feedback_result, tuple) and len(feedback_result) == 2:
                    # μf can override score, diagnostic_text
                    enhanced_score, diagnostic = feedback_result
                    score = float(enhanced_score)
                elif isinstance(feedback_result, dict):
                    # Rich feedback dictionary
                    diagnostic = self._format_rich_feedback(feedback_result, score)
                else:
                    # Simple diagnostic text
                    diagnostic = str(feedback_result)
            except Exception as e:
                logger.warning(f"Enhanced feedback function failed: {e}")
                diagnostic = f"Enhanced feedback failed, score: {score:.2f}"
        else:
            # Default diagnostic when no μf provided
            diagnostic = f"Score: {score:.2f} ({'SUCCESS' if score > 0.5 else 'FAILURE'})"
        
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