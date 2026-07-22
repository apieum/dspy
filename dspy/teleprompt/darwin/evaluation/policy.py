"""Policies controlling which validation examples are evaluated per proposal."""

import random
from typing import Protocol


class EvaluationPolicy(Protocol):
    def get_eval_batch(self, validation_data, *, iteration: int = 0, candidate=None):
        ...


class FullEvaluationPolicy:
    """Evaluate every validation example for every candidate."""

    def get_eval_batch(self, validation_data, *, iteration: int = 0, candidate=None):
        return list(validation_data)


class MinibatchEvaluationPolicy:
    """Evaluate a deterministic validation minibatch per iteration."""

    def __init__(self, batch_size: int, seed: int = 0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.seed = seed

    def get_eval_batch(self, validation_data, *, iteration: int = 0, candidate=None):
        data = list(validation_data)
        if len(data) <= self.batch_size:
            return data
        rng = random.Random(self.seed + iteration)
        return rng.sample(data, self.batch_size)
