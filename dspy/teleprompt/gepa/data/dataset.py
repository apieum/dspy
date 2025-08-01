"""Dataset protocols and implementations for GEPA optimization."""

import logging
from typing import Iterable, List, Protocol

import dspy

logger = logging.getLogger(__name__)


class TrainingDataset(Protocol):
    """Protocol for training datasets used by GEPA.
    
    Follows the paper's approach of splitting dataset into:
    - feedback_data: Used for reflective mutation and feedback collection
    - pareto_data: Used for candidate evaluation and Pareto selection
    """
    
    def feedback_data(self) -> Iterable[dspy.Example]:
        """Generic feedback data for mutation guidance (paper's feedback set)."""
        ...
    
    def pareto_data(self) -> Iterable[dspy.Example]:
        """Evaluation data for candidate selection (paper's Pareto set)."""
        ...


class SplitDataset:
    """Paper-compliant implementation of TrainingDataset.
    
    Splits examples according to pareto_ratio as described in GEPA paper.
    """
    
    def __init__(self, examples: List[dspy.Example], pareto_ratio: float = 0.67):
        """Initialize with examples and split ratio.
        
        Args:
            examples: List of training examples
            pareto_ratio: Fraction of data used for Pareto evaluation (default 0.67 per paper)
        """
        if not 0.0 < pareto_ratio < 1.0:
            raise ValueError(f"pareto_ratio must be between 0 and 1, got {pareto_ratio}")
        
        # Split according to paper's approach
        pareto_count = int(len(examples) * pareto_ratio)
        self._pareto = examples[-pareto_count:] if pareto_count > 0 else []
        self._feedback = examples[:-pareto_count] if pareto_count > 0 else examples
        
        logger.info(f"Split dataset: {len(self._feedback)} feedback examples, {len(self._pareto)} Pareto examples")
    
    def feedback_data(self) -> List[dspy.Example]:
        """Return feedback examples for mutation guidance."""
        return self._feedback
    
    def pareto_data(self) -> List[dspy.Example]:
        """Return Pareto examples for candidate evaluation."""
        return self._pareto