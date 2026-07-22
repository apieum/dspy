"""Optional confidence-aware scoring for assessors exposing logprobs."""

import math
from typing import Any

from .metrics import Metric


class LinearConfidenceScoring:
    """Penalize low-confidence correct answers while preserving binary fallback."""

    def __init__(self, threshold: float = 0.5, minimum: float = 0.3):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if not 0.0 <= minimum < 1.0:
            raise ValueError("minimum must be in [0, 1)")
        self.threshold = threshold
        self.minimum = minimum

    def score(self, value: float, logprob: float | None) -> float:
        if logprob is None or value < 1.0:
            return value
        probability = math.exp(logprob)
        if probability >= self.threshold:
            return value
        ratio = probability / self.threshold
        return self.minimum + (value - self.minimum) * ratio


class ThresholdConfidenceScoring:
    """Require a confidence threshold for a perfect score."""

    def __init__(self, threshold: float = 0.7):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def score(self, value: float, logprob: float | None) -> float:
        if logprob is None or value < 1.0:
            return value
        return value if math.exp(logprob) >= self.threshold else 0.0


def extract_logprob(source: Any) -> float | None:
    """Extract a scalar joint logprob from common DSPy side-info shapes."""
    if isinstance(source, dict):
        for key in ("joint_logprob", "logprob", "log_probability"):
            value = source.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for key in ("side_info", "logprobs"):
            value = extract_logprob(source.get(key))
            if value is not None:
                return value
    for key in ("joint_logprob", "logprob", "log_probability"):
        value = getattr(source, key, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class ConfidenceAssessor:
    """Wrap an assessor with optional logprob-aware score calibration."""

    def __init__(self, assessor, scoring=None):
        self.assessor = assessor
        self.scoring = scoring or LinearConfidenceScoring()

    def __call__(self, example, prediction, trace=None):
        result = self.assessor(example, prediction, trace)
        if not isinstance(result, Metric):
            result = Metric(float(result), trace=trace)
        side_info = result.side_info
        logprob = extract_logprob(side_info)
        if logprob is None:
            logprob = extract_logprob(prediction)
        value = self.scoring.score(float(result.value), logprob)
        enriched = dict(side_info) if isinstance(side_info, dict) else {}
        enriched.update({"logprob": logprob, "confidence_adjusted": value != result.value})
        return Metric(
            value,
            id=result.id,
            feedback=result.feedback,
            errors=result.errors,
            suggestions=result.suggestions,
            trace=result.trace if result.trace is not None else trace,
            objective_scores=result.objective_scores,
            side_info=enriched,
        )
