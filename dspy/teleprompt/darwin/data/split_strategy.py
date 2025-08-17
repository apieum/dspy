"""Data splitting strategies for Darwin."""

import random
import logging
from typing import List, Protocol

import dspy
from ..data.candidate import Candidate

logger = logging.getLogger(__name__)


class DataSplitStrategy(Protocol):
    """Protocol for data splitting strategies."""

    def __init__(self, devset: List[dspy.Example]):
        ...

    def get_feedback_minibatch(self, candidate: Candidate, size: int) -> List[dspy.Example]:
        """Selects examples to guide the generation of a new candidate."""
        ...

    def get_evaluation_minibatch(self, candidate: Candidate, size: int) -> List[dspy.Example]:
        """Selects examples for the fast, initial evaluation of a new candidate."""
        ...


class DefaultSplitStrategy:
    """A simple data splitting strategy that relies on random sampling."""

    def __init__(self, trainset: List[dspy.Example] = None, devset: List[dspy.Example] = None, validation_split: float = 0.2, verbose: bool = False):
        """Initialize the strategy with datasets and validation split ratio.

        Args:
            trainset: Training examples to split into train/validation
            devset: External test set (reserved for final evaluation only)
            validation_split: Fraction of trainset to use for internal validation (default: 0.2)
            verbose: Enable detailed logging
        """
        self.validation_split = validation_split
        self.internal_validation_set = []
        self.external_devset = devset or []
        
        if trainset:
            self._create_splits(trainset, verbose)
    
    def _create_splits(self, trainset: List[dspy.Example], verbose: bool = False) -> None:
        """Create internal train/validation splits from trainset."""
        import random
        
        # Create internal validation set from trainset to prevent data leakage
        trainset_copy = list(trainset)
        random.shuffle(trainset_copy)
        
        validation_size = int(len(trainset_copy) * self.validation_split)
        # Ensure we have at least 1 validation example if trainset is not empty
        if len(trainset_copy) > 0 and validation_size == 0:
            validation_size = 1
        
        self.internal_validation_set = trainset_copy[:validation_size]
        
        if verbose:
            logger.info(f"Split trainset: {len(trainset)} -> training: {len(trainset_copy) - validation_size}, internal validation: {validation_size}")
            logger.info(f"External test set (devset): {len(self.external_devset)} examples - RESERVED for final evaluation only")

    def get_feedback_minibatch(self, candidate: Candidate, size: int) -> List[dspy.Example]:
        """Returns a random minibatch from the INTERNAL validation set for feedback."""
        if not self.internal_validation_set:
            return []

        actual_size = min(size, len(self.internal_validation_set))
        return random.sample(self.internal_validation_set, actual_size)

    def get_evaluation_minibatch(self, candidate: Candidate, size: int) -> List[dspy.Example]:
        """Returns a random minibatch from the INTERNAL validation set for evaluation."""
        if not self.internal_validation_set:
            return []

        actual_size = min(size, len(self.internal_validation_set))
        return random.sample(self.internal_validation_set, actual_size)
    
    def get_final_evaluation_set(self) -> List[dspy.Example]:
        """Returns the external devset for final evaluation ONLY."""
        return self.external_devset
