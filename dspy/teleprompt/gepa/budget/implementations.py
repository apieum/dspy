"""Concrete budget tracker implementations."""

from dataclasses import dataclass
from typing import Dict, Optional

from .base import BudgetTracker


@dataclass
class LLMCallsBudget(BudgetTracker):
    """Budget based on total LLM calls limit."""
    limit: int
    used: int = 0
    minibatch_rollouts: int = 0
    reflection_rollouts: int = 0
    pareto_rollouts: int = 0
    iteration_rollouts: int = 0
    current_iteration: int = 0

    def add_minibatch_cost(self, cost: int):
        self.minibatch_rollouts += cost
        self.used += cost

    def add_reflection_cost(self, cost: int = 1):
        self.reflection_rollouts += cost
        self.used += cost

    def add_pareto_cost(self, cost: int):
        self.pareto_rollouts += cost
        self.used += cost
    
    def add_iteration_cost(self, cost: int = 1):
        """Add cost for iteration overhead."""
        self.iteration_rollouts += cost
        self.used += cost

    def warn_iteration_start(self, iteration: int):
        """Notify budget tracker that new iteration is starting."""
        self.current_iteration = iteration

    def has_budget(self) -> bool:
        """Check if budget allows continuing based on LLM calls limit."""
        return self.used < self.limit

    def get_stats(self) -> Dict[str, int]:
        return {
            'total_used': self.used,
            'minibatch': self.minibatch_rollouts,
            'reflection': self.reflection_rollouts,
            'pareto': self.pareto_rollouts,
            'iteration': self.iteration_rollouts,
            'remaining': self.limit - self.used
        }


@dataclass
class IterationBudget(BudgetTracker):
    """Budget based on maximum number of iterations."""
    max_iterations: int
    current_iteration: int = 0
    used: int = 0
    minibatch_rollouts: int = 0
    reflection_rollouts: int = 0
    pareto_rollouts: int = 0
    iteration_rollouts: int = 0

    def add_minibatch_cost(self, cost: int):
        self.minibatch_rollouts += cost
        self.used += cost

    def add_reflection_cost(self, cost: int = 1):
        self.reflection_rollouts += cost
        self.used += cost

    def add_pareto_cost(self, cost: int):
        self.pareto_rollouts += cost
        self.used += cost
    
    def add_iteration_cost(self, cost: int = 1):
        self.iteration_rollouts += cost
        self.used += cost

    def warn_iteration_start(self, iteration: int):
        self.current_iteration = iteration

    def has_budget(self) -> bool:
        """Check if budget allows continuing based on iteration limit."""
        return self.current_iteration < self.max_iterations

    def get_stats(self) -> Dict[str, int]:
        return {
            'total_used': self.used,
            'minibatch': self.minibatch_rollouts,
            'reflection': self.reflection_rollouts,
            'pareto': self.pareto_rollouts,
            'iteration': self.iteration_rollouts,
            'remaining': self.max_iterations - self.current_iteration
        }


@dataclass
class CombinedBudget(BudgetTracker):
    """Budget that combines both LLM calls and iteration limits."""
    llm_limit: int
    max_iterations: int
    current_iteration: int = 0
    used: int = 0
    minibatch_rollouts: int = 0
    reflection_rollouts: int = 0
    pareto_rollouts: int = 0
    iteration_rollouts: int = 0

    def add_minibatch_cost(self, cost: int):
        self.minibatch_rollouts += cost
        self.used += cost

    def add_reflection_cost(self, cost: int = 1):
        self.reflection_rollouts += cost
        self.used += cost

    def add_pareto_cost(self, cost: int):
        self.pareto_rollouts += cost
        self.used += cost
    
    def add_iteration_cost(self, cost: int = 1):
        self.iteration_rollouts += cost
        self.used += cost

    def warn_iteration_start(self, iteration: int):
        self.current_iteration = iteration

    def has_budget(self) -> bool:
        """Check if budget allows continuing based on both limits."""
        if self.used >= self.llm_limit:
            return False
        return self.current_iteration < self.max_iterations

    def get_stats(self) -> Dict[str, int]:
        return {
            'total_used': self.used,
            'minibatch': self.minibatch_rollouts,
            'reflection': self.reflection_rollouts,
            'pareto': self.pareto_rollouts,
            'iteration': self.iteration_rollouts,
            'remaining_calls': self.llm_limit - self.used,
            'remaining_iterations': self.max_iterations - self.current_iteration
        }