"""CompilationObserver protocol for lifecycle event notifications."""

from typing import List, Protocol, TYPE_CHECKING
import dspy

if TYPE_CHECKING:
    from .data.candidate_pool import CandidatePool
    from .data.cohort import Cohort
    from .budget import Budget


class CompilationObserver(Protocol):
    """Base class for components that observe compilation lifecycle events.
    
    Provides default no-op implementations so components can opt into
    only the lifecycle events they care about.
    """
    
    def start_compilation(self, student: dspy.Module, training_data: List[dspy.Example]) -> None:
        """Called when compilation begins. Components can prepare resources.
        
        Args:
            student: The initial program being optimized
            training_data: The training dataset for optimization
        """
        pass
    
    def finish_compilation(self, result: dspy.Module, final_pool: "CandidatePool") -> None:
        """Called when compilation ends. Components can cleanup/log results.
        
        Args:
            result: The final optimized program
            final_pool: The final candidate pool with all explored candidates
        """
        pass
    
    def start_iteration(self, iteration: int, cohort: "Cohort", budget: "Budget") -> None:
        """Called at start of each optimization iteration.
        
        Args:
            iteration: Current iteration number (0-based)
            cohort: Cohort being processed in this iteration
            budget: Current budget state
        """
        pass
    
    def finish_iteration(self, iteration: int, filtered_cohort: "Cohort", budget: "Budget") -> None:
        """Called after each optimization iteration completes.
        
        Args:
            iteration: Completed iteration number (0-based)
            filtered_cohort: Cohort that survived filtering and evaluation
            budget: Updated budget state after iteration
        """
        pass