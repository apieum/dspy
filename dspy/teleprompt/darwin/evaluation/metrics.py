"""Evaluation metrics for Darwin optimization framework.

This module provides assessors and metrics that work seamlessly with the Darwin evaluation system.
- Assessor: Callable that measures/evaluates predictions
- Metric: The actual measurement result with value and optional feedback
"""

from typing import Dict, Any, List, Union, Callable, Optional
import dspy
import uuid

# Type alias for things that measure/assess predictions
Assessor = Callable[[dspy.Example, Any, Optional[Any]], 'Metric']


class Metric:
    """A metric is a measurement result with value and optional enhanced feedback."""

    def __init__(self,
                 value: Union[float, bool],
                 id: str = "",
                 feedback: str = "",
                 errors: Dict[str, Any] = dict(),
                 suggestions: List[str] = list(),
                 trace: Any = None,
                 objective_scores: Optional[Dict[str, float]] = None):
        """Initialize metric result.

        Args:
            value: The numeric/boolean score
            feedback: Optional textual feedback about the evaluation
            errors: Optional detailed error analysis
            suggestions: Optional list of improvement suggestions
            trace: Optional execution trace data
        """
        if id == "":
            id = str(uuid.uuid4())
        self._id = id
        self.value = value
        self.feedback = feedback
        self.errors = errors
        self.suggestions = suggestions
        self.trace = trace
        self.objective_scores = objective_scores or {}

    @property
    def id(self):
        return self._id

    def __float__(self) -> float:
        """Allow metric to be used as numeric value."""
        return float(self.value)

    def __bool__(self) -> bool:
        """Allow metric to be used as boolean."""
        return bool(self.value)

    def __eq__(self, other) -> bool:
        """Compare metrics by value."""
        if isinstance(other, Metric):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other) -> bool:
        """Compare metrics by value."""
        if isinstance(other, Metric):
            return self.value < other.value
        return self.value < other

    def __str__(self) -> str:
        """String representation showing feedback (or value if no feedback)."""
        return self.feedback if self.feedback else str(self.value)

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Metric(value={self.value}, feedback='{self.feedback[:50]}{'...' if len(self.feedback) > 50 else ''}')"


class BaseAssessor:
    """Base class for all evaluation assessors.

    Provides common functionality for prediction normalization.
    Subclasses implement _evaluate() with normalized inputs and return Metric objects.
    """

    def __call__(self, example: dspy.Example, pred: Any, trace=None) -> Metric:
        """Evaluate prediction against example.

        Args:
            example: DSPy example with expected outputs
            pred: Prediction from model
            trace: Optional execution trace

        Returns:
            Metric object with value and optional feedback
        """
        try:
            normalized_pred = self._normalize_prediction(pred)
            return self._evaluate(example, normalized_pred, trace)
        except Exception as e:
            return Metric(value=0.0, id=getattr(example, "dspy_uuid", ""), feedback="Evaluation failed due to error", errors = {str(__name__): e}, trace=trace)

    def _normalize_prediction(self, pred: Any) -> str:
        """Normalize prediction to consistent string format."""
        if hasattr(pred, 'completions'):
            return pred.completions[0] if pred.completions else ""
        elif hasattr(pred, 'answer'):
            return str(pred.answer)
        elif isinstance(pred, str):
            return pred
        else:
            return str(pred)

    def _get_expected_output(self, example: dspy.Example) -> str:
        """Extract expected output from example."""
        expected = getattr(example, 'answer', None)
        if expected is None:
            expected = getattr(example, 'output', getattr(example, 'label', None))
        return str(expected) if expected is not None else ""

    def _evaluate(self, example: dspy.Example, prediction: str, trace=None) -> Metric:
        """Evaluate normalized prediction against example.

        Args:
            example: DSPy example with expected outputs
            prediction: Normalized prediction string
            trace: Optional execution trace

        Returns:
            Metric object with value and optional feedback
        """
        # Default implementation just returns score without feedback
        score = self._compute_score(example, prediction, trace)
        return Metric(value=score, id=getattr(example, "dspy_uuid", ""), trace=trace)

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> Union[float, bool]:
        """Compute the raw score. Override this method in subclasses.

        Args:
            example: DSPy example with expected outputs
            prediction: Normalized prediction string
            trace: Optional execution trace

        Returns:
            Raw score (float or bool depending on assessor type)
        """
        raise NotImplementedError("Subclasses must implement _compute_score")

    # Comparison magic methods for metric comparison
    def __eq__(self, other) -> bool:
        """Check if two metrics are equivalent."""
        return isinstance(other, self.__class__)

    def __hash__(self) -> int:
        """Allow metrics to be used as dictionary keys."""
        return hash(self.__class__.__name__)

    def __str__(self) -> str:
        """String representation of metric."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}()"


class ExactMatch(BaseAssessor):
    """Exact string matching evaluation.

    Returns True if prediction exactly matches expected answer, False otherwise.
    Case-insensitive comparison after stripping whitespace.

    Best for:
    - Question answering with definitive answers
    - Classification tasks
    - Tasks requiring precise outputs
    """

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> bool:
        expected = self._get_expected_output(example)
        if not expected:
            return False

        pred_str = prediction.strip().lower()
        expected_str = expected.strip().lower()
        return pred_str == expected_str


class Contains(BaseAssessor):
    """Substring containment evaluation.

    Returns True if prediction contains expected answer or vice versa.
    More lenient than exact match for partial credit scenarios.

    Best for:
    - Long-form text generation where exact match is too strict
    - Tasks where the answer might be embedded in longer text
    - Information extraction tasks
    """

    def __init__(self, case_sensitive: bool = False):
        """Initialize contains metric.

        Args:
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.case_sensitive = case_sensitive

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> bool:
        expected = self._get_expected_output(example)
        if not expected:
            return False

        pred_str = prediction
        expected_str = expected

        if not self.case_sensitive:
            pred_str = pred_str.lower()
            expected_str = expected_str.lower()

        return expected_str in pred_str or pred_str in expected_str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(case_sensitive={self.case_sensitive})"


class F1Score(BaseAssessor):
    """F1 score based on token-level overlap.

    Computes harmonic mean of precision and recall using token overlap.
    Provides gradual feedback between 0.0 and 1.0.

    Best for:
    - General-purpose evaluation across many NLP tasks
    - Tasks where both precision and recall matter
    - Question answering where partial credit is desired
    - Text generation with flexible matching
    """

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> float:
        expected = self._get_expected_output(example)
        if not expected:
            return 0.0

        pred_tokens = set(prediction.strip().lower().split())
        expected_tokens = set(expected.strip().lower().split())

        if not pred_tokens and not expected_tokens:
            return 1.0
        if not pred_tokens or not expected_tokens:
            return 0.0

        intersection = pred_tokens.intersection(expected_tokens)
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(expected_tokens) if expected_tokens else 0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)


class RougeL(BaseAssessor):
    """ROUGE-L evaluation using longest common subsequence.

    Measures text similarity based on longest common subsequence (LCS).
    Captures sentence-level structure better than token overlap alone.

    Best for:
    - Text summarization
    - Long-form text generation
    - Tasks where word order and structure matter
    - Document generation and editing
    """

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> float:
        expected = self._get_expected_output(example)
        if not expected:
            return 0.0

        pred_str = prediction.strip()
        expected_str = expected.strip()

        lcs_length = self._lcs_length(pred_str.split(), expected_str.split())

        pred_len = len(pred_str.split())
        exp_len = len(expected_str.split())

        if pred_len == 0 and exp_len == 0:
            return 1.0
        if pred_len == 0 or exp_len == 0:
            return 0.0

        precision = lcs_length / pred_len
        recall = lcs_length / exp_len

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1].lower() == seq2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]


class Bleu(BaseAssessor):
    """BLEU score evaluation with brevity penalty.

    Originally designed for machine translation. Measures n-gram precision
    with penalty for outputs shorter than reference.

    Best for:
    - Machine translation tasks
    - Text generation where precision is more important than recall
    - Tasks with reference translations or canonical outputs
    - Comparing multiple generated variants
    """

    def __init__(self, n_gram: int = 4):
        """Initialize BLEU metric.

        Args:
            n_gram: Maximum n-gram order for BLEU calculation (default: 4)
        """
        self.n_gram = n_gram

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> float:
        expected = self._get_expected_output(example)
        if not expected:
            return 0.0

        pred_tokens = prediction.strip().lower().split()
        ref_tokens = expected.strip().lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Simplified BLEU-1 implementation
        common = 0
        for token in pred_tokens:
            if token in ref_tokens:
                common += 1

        precision = common / len(pred_tokens) if pred_tokens else 0

        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0

        return bp * precision

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_gram={self.n_gram})"


class CustomMetric(BaseAssessor):
    """Wrapper for custom metric functions.

    Allows integration of any custom evaluation function that takes
    (example, prediction) and returns a numeric score.

    Best for:
    - Domain-specific evaluation criteria
    - Task-specific scoring functions
    - Integration with existing evaluation frameworks
    - Complex multi-criteria evaluation
    """

    def __init__(self, metric_func, name: str = "custom"):
        """Initialize custom metric.

        Args:
            metric_func: Function that takes (example, prediction) -> score
            name: Name for the metric (for logging/debugging)
        """
        self.metric_func = metric_func
        self.name = name

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> Union[float, bool]:
        return self.metric_func(example, prediction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

class CompositeMetric(BaseAssessor):
    """Metric that combines multiple metrics with optional weighting.

    Computes weighted average of multiple evaluation metrics.
    Useful for multi-criteria optimization or balanced evaluation.

    Best for:
    - Tasks requiring multiple evaluation aspects
    - Balancing different quality measures
    - A/B testing different metric combinations
    - Multi-objective optimization
    """

    def __init__(self, assessors: Dict[str, Assessor], weights: Dict[str, float] = None):
        """Initialize composite metric.

        Args:
            assessors: Dictionary of assessor_name -> Assessor
            weights: Optional weights for each metric (defaults to equal)
        """
        self.assessors = assessors
        self.weights = weights or {name: 1.0 / len(assessors) for name in assessors}

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {name: w / total_weight for name, w in self.weights.items()}

    def _compute_score(self, example: dspy.Example, prediction: str, trace=None) -> float:
        """Calculate weighted combination of all metrics."""
        total_score = 0.0

        for name, assessor in self.assessors.items():
            score = float(assessor(example, prediction, trace))
            total_score += score * self.weights.get(name, 0.0)

        return total_score

    def _evaluate(self, example: dspy.Example, prediction: str, trace=None) -> Metric:
        objective_scores = {
            name: float(assessor(example, prediction, trace))
            for name, assessor in self.assessors.items()
        }
        score = sum(
            value * self.weights.get(name, 0.0)
            for name, value in objective_scores.items()
        )
        return Metric(
            value=score,
            id=getattr(example, "dspy_uuid", ""),
            trace=trace,
            objective_scores=objective_scores,
        )

    def __repr__(self) -> str:
        metric_names = list(self.assessors.keys())
        return f"{self.__class__.__name__}(metrics={metric_names})"
