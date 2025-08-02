"""Iterations budget implementation."""

from .budget import Budget


class IterationBudget(Budget):
    """Budget that limits by number of iterations."""
    
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
    def can_afford(self, cost: int, operation_type: str) -> bool:
        return self.current_iteration < self.max_iterations
        
    def consume(self, cost: int, operation_type: str) -> None:
        if operation_type == "iteration":
            self.current_iteration += 1
            
    def is_empty(self) -> bool:
        return self.current_iteration >= self.max_iterations
        
    def get_remaining(self) -> int:
        return max(0, self.max_iterations - self.current_iteration)