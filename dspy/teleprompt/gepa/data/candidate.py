"""Candidate data structure for GEPA optimization."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import dspy
from dspy import Module


@dataclass
class Candidate:
    """A candidate solution that encapsulates a DSPy module.
    
    The candidate knows its lineage (parent relationships) but delegates
    all evolutionary operations to the optimization components.
    """
    module: Module  # Pure DSPy module - the actual program
    parents: List['Candidate'] = field(default_factory=list)  # Direct parent references
    generation_number: int = 0  # Which generation this belongs to
    creation_timestamp: float = field(default_factory=time.time)
    fitness_history: List[float] = field(default_factory=list)  # Performance over time
    task_scores: List[float] = field(default_factory=list)  # task_scores[task_id] = score
    
    def __hash__(self) -> int:
        """Make candidates hashable based on object identity."""
        return hash(id(self))
    
    def __eq__(self, other) -> bool:
        """Compare candidates by object identity."""
        if not isinstance(other, Candidate):
            return False
        return self is other
    
    def add_fitness_score(self, score: float) -> None:
        """Record a new fitness evaluation."""
        self.fitness_history.append(score)
    
    def average_fitness(self) -> float:
        """Calculate average fitness across all evaluations."""
        return sum(self.fitness_history) / len(self.fitness_history) if self.fitness_history else 0.0
    
    def set_task_score(self, task_id: int, score: float) -> None:
        """Set score for a specific task during evaluation."""
        # Extend the list to accommodate this task_id
        while len(self.task_scores) <= task_id:
            self.task_scores.append(0.0)
        
        self.task_scores[task_id] = score
    
    def set_task_scores(self, task_scores: Dict[int, float]) -> None:
        """Set scores for multiple tasks."""
        for task_id, score in task_scores.items():
            self.set_task_score(task_id, score)
    
    def task_score(self, task_id: int) -> Optional[float]:
        """Get score for a specific task."""
        return self.task_scores[task_id] if task_id < len(self.task_scores) else None
    
    def average_task_score(self) -> float:
        """Calculate average score across all tasks."""
        return sum(self.task_scores) / len(self.task_scores) if self.task_scores else 0.0
    
    def evaluate_on_example(self, example: dspy.Example, metric: Callable) -> float:
        """Evaluate this candidate on a single example using provided metric."""
        try:
            prediction = self.module(**example.inputs())
            score = metric(example, prediction)
            return float(score)
        except Exception:
            return 0.0  # Failed evaluation
    
    def evaluate_on_batch(self, examples: List[dspy.Example], metric: Callable) -> float:
        """Evaluate this candidate on a batch of examples."""
        if not examples:
            return 0.0
        
        total_score = sum(self.evaluate_on_example(ex, metric) for ex in examples)
        avg_score = total_score / len(examples)
        self.add_fitness_score(avg_score)
        return avg_score
    
    def best_for_task(self, task_id: int, other: 'Candidate') -> 'Candidate':
        """Compare two candidates and return the best one for a specific task.
        
        Considers both score and generation number (newer wins on ties).
        
        Args:
            task_id: The task to compare performance on
            other: The other candidate to compare against
            
        Returns:
            The better candidate for the task (self or other)
        """
        my_score = self.task_score(task_id) or 0.0
        other_score = other.task_score(task_id) or 0.0
        
        # Better score wins
        if my_score > other_score:
            return self
        
        # Same score - newer generation wins
        if my_score == other_score and self.generation_number > other.generation_number:
            return self
            
        return other