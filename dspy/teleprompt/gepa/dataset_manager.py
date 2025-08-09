"""DatasetManager architecture for centralized dataset management in GEPA."""

from typing import List, Dict, Optional, Protocol, Callable, Any
import random
import dspy


class DatasetManager(Protocol):
    """Protocol for dataset management in GEPA optimization."""

    @property
    def num_eval_tasks(self) -> int:
        """Get the number of tasks in the evaluation dataset."""
        ...

    def get_eval_set(self) -> Dict[Any, dspy.Example]:
        """Get the full evaluation dataset for evaluation."""
        ...

    def get_validation_minibatch(self, size: int) -> Dict[Any, dspy.Example]:
        """Get a minibatch from dev data for validation."""
        ...

    def get_feedback_minibatch(self, size: int) -> Dict[Any, dspy.Example]:
        """Get a minibatch from dev data for generation."""
        ...


class DatasetManagerFactory(Protocol):
    """Factory protocol for creating DatasetManager instances."""

    def create(self, training_data: List[dspy.Example]|Dict[Any, dspy.Example]) -> DatasetManager:
        """Create a DatasetManager instance from training data."""
        ...


class DefaultDatasetManager:
    """Default implementation of dataset management for GEPA."""

    def __init__(self, training_data: List[dspy.Example]|Dict[Any, dspy.Example], split_ratio: float = 0.2):
        """Initialize dataset manager with training data.

        Args:
            training_data: Full training dataset
            split_ratio: Fraction of data to use for evaluation
        """
        training_data = training_data.copy()
        if isinstance(training_data, list):
            training_data = {task_id: task for (task_id, task) in enumerate(training_data)}

        # Handle edge case for small datasets
        if len(training_data) < 2:
            # Use same data for both datasets
            self.eval_data = training_data.copy()
            self.dev_data = training_data.copy()
        else:
            # Normal split
            nsplit = min(max(1, int(len(training_data) * split_ratio)), len(training_data) - 1)
            it = list(training_data.items())
            self.eval_data = dict(it[:nsplit])
            self.dev_data = dict(it[nsplit:])

    @property
    def num_eval_tasks(self) -> int:
        """Get the number of tasks in the eval dataset."""
        return len(self.eval_data)

    @property
    def num_dev_examples(self) -> int:
        """Get the number of examples in the dev dataset."""
        return len(self.dev_data)

    def get_eval_set(self) -> Dict[Any, dspy.Example]:
        """Get the full eval dataset for evaluation."""
        return self.eval_data.copy()

    def get_validation_minibatch(self, size: int) -> Dict[Any, dspy.Example]:
        """Get a minibatch from feedback data for validation."""
        if not self.dev_data:
            return {}

        actual_size = min(size, len(self.dev_data))
        return dict(random.sample(list(self.dev_data.items()), actual_size))

    def get_feedback_minibatch(self, size: int) -> Dict[Any, dspy.Example]:
        """Get a minibatch from feedback data for generation."""
        if not self.dev_data:
            return {}

        actual_size = min(size, len(self.dev_data))
        return dict(random.sample(list(self.dev_data.items()), actual_size))


class DefaultDatasetManagerFactory:
    """Default factory for creating DefaultDatasetManager instances."""

    def __init__(self, split_ratio: float = 0.2):
        """Initialize factory with configuration.

        Args:
            split_ratio: Fraction of data to use for evaluation
        """
        self.split_ratio = split_ratio

    def create(self, training_data: List[dspy.Example]) -> DefaultDatasetManager:
        """Create a DatasetManager instance from training data."""
        return DefaultDatasetManager(training_data, self.split_ratio)
