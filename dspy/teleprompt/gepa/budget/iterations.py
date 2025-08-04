"""Iterations budget implementation."""

from typing import Dict, Any, Optional
import dspy
from .budget import Budget


class IterationBudget(Budget):
    """Budget that limits by number of iterations."""
    
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
        
    def get_remaining(self) -> dict:
        remaining_iterations = max(0, self.max_iterations - self.current_iteration)
        return {
            "iterations": remaining_iterations,
            "percentage": (remaining_iterations / self.max_iterations) * 100 if self.max_iterations > 0 else 0
        }
    
    
    def start_iteration(self, iteration: int, cohort, budget) -> None:
        """Track iteration start - increment iteration counter."""
        self.current_iteration = iteration