"""DatasetManager architecture for centralized dataset management in GEPA."""

from typing import List, Optional, Protocol, Callable
import random
import dspy


class DatasetManager(Protocol):
    """Protocol for dataset management in GEPA optimization."""
    
    @property
    def num_pareto_tasks(self) -> int:
        """Get the number of tasks in the pareto dataset."""
        ...
    
    def get_pareto_set(self) -> List[dspy.Example]:
        """Get the full pareto dataset for evaluation."""
        ...
    
    def get_validation_minibatch(self, size: int) -> List[dspy.Example]:
        """Get a minibatch from feedback data for validation."""
        ...
    
    def get_feedback_minibatch(self, size: int) -> List[dspy.Example]:
        """Get a minibatch from feedback data for generation."""
        ...


class DatasetManagerFactory(Protocol):
    """Factory protocol for creating DatasetManager instances."""
    
    def create(self, training_data: List[dspy.Example]) -> DatasetManager:
        """Create a DatasetManager instance from training data."""
        ...


class DefaultDatasetManager:
    """Default implementation of dataset management for GEPA."""
    
    def __init__(self, training_data: List[dspy.Example], pareto_split_ratio: float = 0.2):
        """Initialize dataset manager with training data.
        
        Args:
            training_data: Full training dataset
            pareto_split_ratio: Fraction of data to use for pareto evaluation
        """
        self.training_data = training_data.copy()
        
        # Handle edge case for small datasets
        if len(training_data) < 2:
            # Use same data for both datasets
            self.pareto_data = training_data.copy()
            self.feedback_data = training_data.copy()
        else:
            # Normal split
            npareto = min(max(1, int(len(training_data) * pareto_split_ratio)), len(training_data) - 1)
            self.pareto_data = training_data[:npareto]
            self.feedback_data = training_data[npareto:]
    
    @property
    def num_pareto_tasks(self) -> int:
        """Get the number of tasks in the pareto dataset."""
        return len(self.pareto_data)
    
    @property
    def num_feedback_examples(self) -> int:
        """Get the number of examples in the feedback dataset."""
        return len(self.feedback_data)
    
    def get_pareto_set(self) -> List[dspy.Example]:
        """Get the full pareto dataset for evaluation."""
        return self.pareto_data.copy()
    
    def get_validation_minibatch(self, size: int) -> List[dspy.Example]:
        """Get a minibatch from feedback data for validation."""
        if not self.feedback_data:
            return []
        
        actual_size = min(size, len(self.feedback_data))
        return random.sample(self.feedback_data, actual_size)
    
    def get_feedback_minibatch(self, size: int) -> List[dspy.Example]:
        """Get a minibatch from feedback data for generation."""
        if not self.feedback_data:
            return []
            
        actual_size = min(size, len(self.feedback_data))
        return random.sample(self.feedback_data, actual_size)


class DefaultDatasetManagerFactory:
    """Default factory for creating DefaultDatasetManager instances."""
    
    def __init__(self, pareto_split_ratio: float = 0.2):
        """Initialize factory with configuration.
        
        Args:
            pareto_split_ratio: Fraction of data to use for pareto evaluation
        """
        self.pareto_split_ratio = pareto_split_ratio
    
    def create(self, training_data: List[dspy.Example]) -> DefaultDatasetManager:
        """Create a DatasetManager instance from training data."""
        return DefaultDatasetManager(training_data, self.pareto_split_ratio)