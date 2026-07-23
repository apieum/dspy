"""Batch-evaluation strategies for candidate programs."""

from abc import ABC, abstractmethod


class BatchEvaluator(ABC):
    """Evaluate candidate/example jobs through a replaceable strategy."""

    @abstractmethod
    def evaluate(self, jobs, assessor, channel):
        """Return one score list for each ``(candidate, examples)`` job."""
        raise NotImplementedError


class PerCandidateBatchEvaluator(BatchEvaluator):
    """Default: batch examples within each candidate, without cross-candidate batching."""

    def evaluate(self, jobs, assessor, channel):
        return [
            candidate.evaluate_on_batch(
                examples,
                assessor=assessor,
                channel=channel,
                persist_scores=False,
            )
            for candidate, examples in jobs
        ]


class CallbackBatchEvaluator(BatchEvaluator):
    """Adapter for an external batch-evaluation callback."""

    def __init__(self, callback):
        if not callable(callback):
            raise TypeError("batch evaluator callback must be callable")
        self.callback = callback

    def evaluate(self, jobs, assessor, channel):
        return self.callback(jobs, assessor, channel)


def resolve_batch_evaluator(value) -> BatchEvaluator:
    """Normalize configuration once, outside the evaluation hot path."""
    if value is None:
        raise ValueError("batch_evaluator must be configured")
    if isinstance(value, BatchEvaluator):
        return value
    if callable(value):
        return CallbackBatchEvaluator(value)
    raise TypeError("batch_evaluator must be a BatchEvaluator or callable callback")
