"""Centralized dataset management for Darwin strategies."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Protocol

import dspy


class DatasetManager(Protocol):
    """Interface used by strategies to access evaluation data."""

    @property
    def num_eval_tasks(self) -> int: ...

    @property
    def num_dev_examples(self) -> int: ...

    def get_eval_set(self) -> Dict[Any, dspy.Example]: ...

    def get_validation_minibatch(self, size: int) -> Dict[Any, dspy.Example]: ...

    def get_feedback_minibatch(self, size: int) -> Dict[Any, dspy.Example]: ...


class DatasetManagerFactory(Protocol):
    """Factory for creating a dataset manager at compilation time."""

    def create(
        self,
        training_data: List[dspy.Example] | Dict[Any, dspy.Example],
        validation_data: Optional[List[dspy.Example] | Dict[Any, dspy.Example]] = None,
    ) -> DatasetManager: ...


class DefaultDatasetManager:
    """Keep training/evaluation and development data in one stable interface."""

    def __init__(
        self,
        training_data: List[dspy.Example] | Dict[Any, dspy.Example],
        validation_data: Optional[List[dspy.Example] | Dict[Any, dspy.Example]] = None,
        split_ratio: float = 0.2,
        seed: Optional[int] = None,
    ):
        self._rng = random.Random(seed)
        training = self._as_dict(training_data)
        if validation_data is not None:
            self.eval_data = training
            self.dev_data = self._as_dict(validation_data)
            return

        if len(training) < 2:
            self.eval_data = training.copy()
            self.dev_data = training.copy()
            return

        split_size = min(max(1, int(len(training) * split_ratio)), len(training) - 1)
        items = list(training.items())
        self.eval_data = dict(items[:split_size])
        self.dev_data = dict(items[split_size:])

    @staticmethod
    def _as_dict(
        data: List[dspy.Example] | Dict[Any, dspy.Example],
    ) -> Dict[Any, dspy.Example]:
        if isinstance(data, dict):
            return data.copy()
        return {task_id: example for task_id, example in enumerate(data)}

    @property
    def num_eval_tasks(self) -> int:
        return len(self.eval_data)

    @property
    def num_dev_examples(self) -> int:
        return len(self.dev_data)

    def get_eval_set(self) -> Dict[Any, dspy.Example]:
        return self.eval_data.copy()

    def _sample_dev(self, size: int) -> Dict[Any, dspy.Example]:
        if not self.dev_data or size <= 0:
            return {}
        count = min(size, len(self.dev_data))
        return dict(self._rng.sample(list(self.dev_data.items()), count))

    def get_validation_minibatch(self, size: int) -> Dict[Any, dspy.Example]:
        return self._sample_dev(size)

    def get_feedback_minibatch(self, size: int) -> Dict[Any, dspy.Example]:
        return self._sample_dev(size)


class DefaultDatasetManagerFactory:
    """Default factory using a configurable validation split."""

    def __init__(self, split_ratio: float = 0.2, seed: Optional[int] = None):
        self.split_ratio = split_ratio
        self.seed = seed

    def create(
        self,
        training_data: List[dspy.Example] | Dict[Any, dspy.Example],
        validation_data: Optional[List[dspy.Example] | Dict[Any, dspy.Example]] = None,
    ) -> DefaultDatasetManager:
        return DefaultDatasetManager(training_data, validation_data, self.split_ratio, self.seed)
