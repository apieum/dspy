"""Mutation generation implementation."""

import random
from typing import List
import dspy
from .generator import Generator
from ..data.candidate import Candidate
from .generation import Generation


class MutationGenerator(Generator):
    """Generator using mutation of existing candidates."""
    
    def __init__(self, mutation_rate: float = 0.3, population_size: int = 10):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        
    def generate(self, parent_candidates: List[Candidate], 
                feedback_data: List[dspy.Example], 
                iteration: int) -> Generation:
        
        new_candidates = []
        
        for i in range(self.population_size):
            if parent_candidates and random.random() < self.mutation_rate:
                # Select random parent for mutation
                parent = random.choice(parent_candidates)
                
                # Create mutated copy (simplified - real implementation would use proper mutation)
                mutated_module = parent.module.deepcopy()
                
                new_candidate = Candidate(
                    module=mutated_module,
                    parent_ids=[parent.candidate_id],
                    generation_number=iteration
                )
                new_candidates.append(new_candidate)
                
        return Generation(
            candidates=new_candidates,
            generation_id=iteration,
            iteration=iteration
        )