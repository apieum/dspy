"""Composite generator for combining multiple generation strategies."""

import logging
from typing import Iterable, List

import dspy

from ..data.candidate_pool import CandidatePool
from .base import CandidateGenerator

logger = logging.getLogger(__name__)


class CompositeGenerator(CandidateGenerator):
    """Composite generator that combines multiple generation strategies.
    
    Allows composition of different generation strategies (mutation, crossover, etc.)
    running them in sequence and collecting all generated candidates.
    """
    
    def __init__(self, generators: List[CandidateGenerator]):
        """Initialize composite generator.
        
        Args:
            generators: List of generators to run in sequence
        """
        if not generators:
            raise ValueError("CompositeGenerator requires at least one generator")
        self.generators = generators
    
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[dspy.Example], 
                          iteration: int) -> List[dspy.Module]:
        """Generate candidates using all constituent generators.
        
        Runs all generators in sequence and combines their results.
        """
        all_candidates = []
        feedback_data_list = list(feedback_data)  # Convert once for reuse
        
        for generator in self.generators:
            try:
                new_candidates = generator.generate_candidates(
                    candidate_pool, feedback_data_list, iteration
                )
                all_candidates.extend(new_candidates)
            except Exception as e:
                logger.warning(f"Generator {type(generator).__name__} failed: {e}")
                continue
        
        return all_candidates