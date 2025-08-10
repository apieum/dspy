"""ReflectivePromptMutation - Evolutionary mutation using reflection on feedback."""

import logging
from typing import Any, List, Optional, TYPE_CHECKING
import random

import dspy
from .generator import Generator
from .reflection_strategy import ReflectionStrategy, GEPAReflection
from .evolvable_module import EvolvableModule
from .prompt_mutator import ReflectivePromptMutator
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns

if TYPE_CHECKING:
    from ..dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class ReflectivePromptMutation(Generator):
    """
    GEPA's reflective mutation, focused purely on generation.
    """

    def __init__(self,
                 feedback_provider,
                 reflection_strategy: Optional[ReflectionStrategy] = None,
                 reflection_lm: Optional[Any] = None,
                 minibatch_size: int = 3,
                 module_selection: str = "round_robin"):
        if feedback_provider is None:
            raise ValueError("ReflectivePromptMutation requires a FeedbackProvider")

        self.feedback_provider = feedback_provider
        self.reflection_strategy = reflection_strategy or GEPAReflection()
        self.reflection_lm = reflection_lm
        self.minibatch_size = minibatch_size
        self.module_selection = module_selection

        self.dataset_manager: Optional["DatasetManager"] = None
        self.next_module_idx = 0

    def start_compilation(self, student: dspy.Module, dataset_manager: "DatasetManager") -> None:
        """Initialize with dataset manager for reflection."""
        self.dataset_manager = dataset_manager
        self.next_module_idx = 0

    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate a new candidate without validation."""
        if parents.is_empty() or not self.dataset_manager:
            if budget:
                budget.spend_on_generation(None, {"type": "no_parents_or_data"})
            return NewBorns()

        try:
            selected_parents = parents.sample_stochastic(1)
            if selected_parents.is_empty():
                if budget:
                    budget.spend_on_generation(None, {"type": "no_selected_parent"})
                return NewBorns()

            parent = list(selected_parents)[0]

            predictors = parent.module.predictors()
            if not predictors:
                if budget:
                    budget.spend_on_generation(None, {"type": "no_predictors"})
                return NewBorns()

            module_idx = self._select_target_module(len(predictors))

            minibatch = self.dataset_manager.get_feedback_minibatch(self.minibatch_size)
            if not minibatch:
                if budget:
                    budget.spend_on_generation(None, {"type": "no_minibatch"})
                return NewBorns()

            evolvable = self._ensure_evolvable(parent.module)
            feedback = evolvable.collect_traces_and_evaluate(
                minibatch, self.feedback_provider, module_idx
            )

            # Evolve using a PromptMutator strategy
            mutator = ReflectivePromptMutator(self.reflection_strategy, self.reflection_lm)
            child_module = mutator.mutate(evolvable, feedback, module_idx)

            child_candidate = Candidate(
                module=child_module,
                generation_number=parent.generation_number + 1,
                parents=[parent],
            )

            # Spend budget for the generation (reflection + mutations)
            if budget:
                budget.spend_on_generation(child_module, {
                    "type": "reflective_mutation",
                    "module_idx": module_idx
                })

            return NewBorns(child_candidate, iteration=parents.iteration)

        except Exception as e:
            logger.warning(f"Reflective prompt mutation failed: {e}")
            if budget:
                budget.spend_on_generation(None, {"type": "failed_mutation", "error": str(e)})
            return NewBorns()

    def _select_target_module(self, num_modules: int) -> int:
        """Select module to mutate."""
        if self.module_selection == "round_robin":
            module_idx = self.next_module_idx % num_modules
            self.next_module_idx = (self.next_module_idx + 1)
            return module_idx
        elif self.module_selection == "random":
            return random.randint(0, num_modules - 1)
        else:
            raise ValueError(f"Unknown module selection strategy: {self.module_selection}")


    def _ensure_evolvable(self, module: dspy.Module) -> EvolvableModule:
        """Wrap DSPy module as EvolvableModule."""
        if isinstance(module, EvolvableModule):
            return module

        # This assumes EvolvableModule can be created from a base module
        return EvolvableModule(base_module=module)
