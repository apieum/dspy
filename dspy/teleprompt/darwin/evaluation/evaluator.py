"""Evaluator protocol for GEPA optimization."""

from abc import abstractmethod
import inspect
from typing import Callable, List, Type
from ..compilation_observer import CompilationObserver
from dspy import Module
from ..data.cohort import Survivors, NewBorns
from ..budget import Budget


class Evaluator(CompilationObserver):
    """Protocol for evaluating and filtering new candidates.

    This component owns a metric and decides which newly generated
    candidates should be promoted (kept) vs discarded. It encapsulates
    the two-phase evaluation logic from the GEPA paper.
    """

    @abstractmethod
    def evaluate(self, cohort: "NewBorns", budget: "Budget") -> "Survivors":
        """
        Evaluates new candidates. If a candidate has parents, it undergoes
        two-phase validation. If it has no parents (the initial candidate),
        it is automatically promoted to full evaluation.

        Args:
            cohort: The cohort of newly generated candidates to evaluate.
            budget: The budget manager to track evaluation costs.

        Returns:
            A Survivors cohort containing only the candidates that were
            successfully promoted after passing evaluation.
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
        # --- Step 1 & 2: Inspect and Combine __init__ Signatures ---
        params = {}
        combined = {}
        for eval_cls in evaluator_classes:
            sig = inspect.signature(eval_cls.__init__)
            params[eval_cls] = []
            for param in sig.parameters.values():
                if param.name not in ("self", "args", "kwargs"):
                    params[eval_cls].append(param.name)
                    combined[param.name] = param

        # Create the new signature object for introspection
        new_signature = inspect.Signature(
            parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(combined.values())
        )

        def __init__(self, *args, **kwargs):
            """Initializes the EvalChain, creating instances of all chained evaluators with their positional arguments"""
            super(self.__class__, self).__init__(*args, **kwargs)
            # Bind the provided arguments to the combined signature to handle both
            # positional and keyword arguments correctly.
            try:
                bound_args = new_signature.bind(self, *args, **kwargs)
            except TypeError as e:
                raise TypeError(f"Error calling {name}.__init__(): {e}") from e

            bound_args.apply_defaults()
            # This dictionary now contains all arguments, correctly mapped.
            all_provided_args = bound_args.arguments
            all_provided_args.pop('self', None)

            self.dataset_manager = None
            self.evaluators = []
            # eval_cls(**kwargs) for eval_cls in evaluator_classes
            for eval_cls in evaluator_classes:
                init_kwargs = { key: value for key, value in all_provided_args.items() if key in params[eval_cls]}
                self.evaluators.append(eval_cls(**init_kwargs))

        # Attach the dynamic signature for introspection tools (help(), IDEs)
        __init__.__signature__ = new_signature

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

        def evaluate(self, cohort: "NewBorns", budget: "Budget") -> "Survivors":
            """Executes the chain of evaluators sequentially."""
            current_cohort = cohort
            survivors = Survivors(*cohort.to_list(), iteration=cohort.iteration)
            for i, evaluator in enumerate(self.evaluators):
                survivors = evaluator.evaluate(current_cohort, budget)

                # If not the last step, convert survivors to newborns for the next evaluator
                if i < len(self.evaluators) - 1:
                    current_cohort = NewBorns(*survivors.to_list(), iteration=survivors.iteration)

            return survivors

        def start_compilation(self, student: Module, dataset_manager) -> None:
            self.dataset_manager = dataset_manager
            for evaluator in self.evaluators:
                evaluator.start_compilation(student, dataset_manager)

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
