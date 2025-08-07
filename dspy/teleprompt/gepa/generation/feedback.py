"""Feedback provider for GEPA optimization."""

import logging
from typing import Any, Callable, List, Optional

import dspy

logger = logging.getLogger(__name__)


class FeedbackProvider:
    """Encapsulates metric (μ) and enhanced feedback function (μf) for GEPA.
    
    This class provides a clean interface for evaluation and diagnostic feedback,
    supporting both simple metrics and rich μf-compliant metrics that return
    detailed evaluation traces for intelligent reflection.
    
    The metric can return either:
    - float: Simple score (backward compatible)
    - (float, str): Score and rich feedback text (μf-compliant)
    """
    
    def __init__(self, metric: Callable, feedback_function: Optional[Callable] = None):
        """Initialize feedback provider.
        
        Args:
            metric: Evaluation function μ that returns either:
                   - float: Simple score for backward compatibility  
                   - (float, str): Score and rich diagnostic text (μf-compliant)
            feedback_function: Optional enhanced feedback function μf for additional diagnostics
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
        # Get score and optional feedback_text using metric (μ/μf)
        feedback_text = None
        try:
            metric_result = self.metric(example, prediction, trace)
            
            # Unpack the result for μf compliance
            if isinstance(metric_result, tuple):
                score, feedback_text = metric_result
            else:
                score = metric_result  # Maintain backward compatibility
                
            score = float(score)
            
        except TypeError:
            # Fallback for metrics that don't accept trace parameter
            metric_result = self.metric(example, prediction)
            if isinstance(metric_result, tuple):
                score, feedback_text = metric_result
            else:
                score = metric_result
            score = float(score)
        
        # Start with base diagnostic from metric evaluation
        status = "SUCCESS" if score > 0.5 else "FAILURE"
        diagnostic = f"Score: {score:.2f} ({status})"
        
        # Append rich feedback from metric's μf output if available
        if feedback_text:
            diagnostic += f" | Evaluator Feedback: {feedback_text}"
        
        # Get additional diagnostic feedback from enhanced feedback function
        if self.feedback_function:
            try:
                # Enhanced feedback function (μf) provides additional rich diagnostics
                feedback_result = self.feedback_function(example, prediction, trace, module_idx)
                
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