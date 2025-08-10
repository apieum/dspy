"""Feedback data structures for evaluation step."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationTrace:
    """Rich evaluation trace as described in paper Section 3.2."""
    execution_steps: List[str]
    compilation_errors: List[str]
    intermediate_outputs: List[Any]
    module_outputs: Dict[int, Any]
    reasoning_chains: List[str]
    tool_calls: List[Dict]
    error_messages: List[str]
    performance_metrics: Dict[str, float]


@dataclass
class ModuleFeedback:
    """Per-module feedback for multi-hop systems."""
    module_id: int
    module_name: str
    input_data: Any
    output_data: Any
    execution_time: float
    success: bool
    error_message: Optional[str]
    intermediate_reasoning: List[str]
    confidence_score: float


@dataclass
class FeedbackResult:
    """Enhanced feedback with traces and diagnostics for reflective mutation."""
    traces: List[List]  # DSPy traces: List of (predictor, inputs, outputs) tuples
    diagnostics: List[str]  # Textual diagnostic feedback
    scores: List[float]  # Scalar scores for each example
    # Enhanced Feedback Function μf fields
    evaluation_traces: List[EvaluationTrace] = None  # Rich evaluation traces
    module_feedback: List[ModuleFeedback] = None  # Module-level feedback
    feedback_text: List[str] = None  # Textual diagnostic feedback
    fitness_history: List[float] = None  # Historical fitness for lineage analysis

    def __post_init__(self):
        if self.evaluation_traces is None:
            self.evaluation_traces = []
        if self.module_feedback is None:
            self.module_feedback = []
        if self.feedback_text is None:
            self.feedback_text = []
        if self.fitness_history is None:
            self.fitness_history = []