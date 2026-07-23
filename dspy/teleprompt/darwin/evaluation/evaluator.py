"""Generic evaluation protocol for Darwin algorithms."""

from abc import abstractmethod
from typing import List, Type, TYPE_CHECKING
import dspy
from dspy import Module
from ..data.cohort import Cohort
from ..budget import Budget
from ..observers import Channel

if TYPE_CHECKING:
    from ..config import DarwinConfig


class Evaluator(Channel):
    """Protocol for evaluating and filtering candidate cohorts.

    This component owns a metric and decides which newly generated
    candidates should be promoted (kept) or discarded. Concrete algorithms
    decide how many evaluation phases they need and which cohort type they
    return.
    """

    def __init__(self):
        """Initialize evaluator with observer support."""
        super().__init__()
        self.dataset_manager = None

    def start_compilation(self, student: dspy.Module, dataset_manager=None, verbose: bool = False) -> None:
        """Attach compilation data without removing the list-based API."""
        self.dataset_manager = dataset_manager
        self.verbose = verbose

    @abstractmethod
    def evaluate(self, cohort: "Cohort", budget: "Budget") -> "Cohort":
        """
        Evaluate a cohort and return the cohort accepted by this phase.

        Args:
            new_borns: The cohort of newly generated candidates to evaluate.
            budget: The budget manager to track evaluation costs.

        Returns:
            A cohort containing candidates accepted by the phase.
        """
        ...

    @classmethod
    def create_chain(cls, name: str, evaluator_classes: List[Type["Evaluator"]]) -> Type["Evaluator"]:
        """
        Dynamically creates a new Evaluator class that chains multiple evaluators.

        The generated class will instantiate and run each evaluator in sequence,
        using the output of one as the input to the next. The new class will
        inherit from the class this method is called on (e.g., `Evaluator`).

        Args:
            name (str): The desired name for the new dynamic Evaluator class.
            evaluator_classes (List[Type[Evaluator]]): A list of Evaluator classes
            (not instances) to be chained together.

        Returns:
            A new class that inherits from the calling class (cls).
        """
        def __init__(self, *, config: "DarwinConfig", **kwargs):
            """Create every evaluator with the same explicit Darwin configuration."""
            cls.__init__(self)
            self.config = config
            self.evaluators = [
                eval_cls(config=config, **kwargs)
                for eval_cls in evaluator_classes
            ]

        # --- Add __getattr__ for delegation ---
        def __getattr__(self, name, default=None):
            """
                Delegate attribute access to the internal evaluators.
                Searches for the attribute in each component evaluator in order.
            """
            for evaluator in self.evaluators:
                if hasattr(evaluator, name):
                    return getattr(evaluator, name)
            return default

        def evaluate(self, cohort: "Cohort", budget: "Budget") -> "Cohort":
            """Executes the chain of evaluators sequentially."""
            current_cohort = cohort
            for evaluator in self.evaluators:
                current_cohort = evaluator.evaluate(current_cohort, budget)
            return current_cohort

        def start_compilation(self, student: dspy.Module, dataset_manager=None, verbose: bool=False) -> None:
            self.dataset_manager = dataset_manager
            for evaluator in self.evaluators:
                evaluator.start_compilation(student, dataset_manager=dataset_manager, verbose=verbose)

        def finish_compilation(self, result: Module) -> None:
            for evaluator in self.evaluators:
                evaluator.finish_compilation(result)

        def start_iteration(self, iteration: int, cohort: "Cohort", budget: "Budget") -> None:
            for evaluator in self.evaluators:
                evaluator.start_iteration(iteration, cohort, budget)

        def finish_iteration(self, iteration: int, filtered_cohort: "Cohort", budget: "Budget") -> None:
            for evaluator in self.evaluators:
                evaluator.finish_iteration(iteration, filtered_cohort, budget)

        class_dict = {
            "__init__": __init__,
            "__getattr__": __getattr__,
            "evaluate": evaluate,
            "start_compilation": start_compilation,
            "finish_compilation": finish_compilation,
            "start_iteration": start_iteration,
            "finish_iteration": finish_iteration,
        }

        # Create the new class, inheriting from `cls`
        return type(name, (cls,), class_dict)
