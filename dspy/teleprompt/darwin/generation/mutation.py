"""ReflectivePromptMutation - Evolutionary mutation using reflection on feedback."""

import logging
from typing import Any, List, Optional, Dict
import random

import dspy
from .generator import Generator
from .reflection_strategy import ReflectionStrategy, GEPAReflection
from .evolvable_module import EvolvableModule
from .prompt_mutator import ReflectivePromptMutator
from .dspy_utils import get_predictors
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns

logger = logging.getLogger(__name__)


class ReflectivePromptMutation(Generator):
    """
    GEPA's reflective mutation, focused purely on generation.
    """

    def __init__(self,
                 feedback_provider,
                 feedback_data: List[dspy.Example] = None,
                 reflection_strategy: Optional[ReflectionStrategy] = None,
                 reflection_lm: Optional[Any] = None,
                 module_selection: str = "round_robin"):
        super().__init__()
        if feedback_provider is None:
            raise ValueError("ReflectivePromptMutation requires a FeedbackProvider")

        self.feedback_provider = feedback_provider
        self.feedback_data = feedback_data or []
        self.reflection_strategy = reflection_strategy or GEPAReflection()
        self.reflection_lm = reflection_lm
        self.module_selection = module_selection

        self.next_module_idx = 0

    def start_compilation(self, student: dspy.Module, verbose: bool=False) -> None:
        """Set verbose mode for the generator."""
        self.next_module_idx = 0
        self.verbose = verbose

    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate a new candidate without validation."""
        if parents.is_empty() or not self.feedback_data:
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

            predictors = get_predictors(parent.module)
            if not predictors:
                if budget:
                    budget.spend_on_generation(None, {"type": "no_predictors"})
                return NewBorns()

            module_idx = self._select_target_module(len(predictors))

            minibatch = self._get_feedback_minibatch()
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
            child_module = mutator.mutate(evolvable, feedback, module_idx, verbose=getattr(self, 'verbose', False))

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
            self.publish('mutation_failure', None, {'reason': f'Reflective prompt mutation failed: {e}'})
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


    def _get_feedback_minibatch(self) -> Dict[int, dspy.Example]:
        """Get the feedback data provided by strategy."""
        if not self.feedback_data:
            return {}

        # Use all feedback data provided by strategy (strategy already split it appropriately)
        return {i: example for i, example in enumerate(self.feedback_data)}

    def _ensure_evolvable(self, module: dspy.Module) -> EvolvableModule:
        """Wrap DSPy module as EvolvableModule."""
        if isinstance(module, EvolvableModule):
            return module

        # This assumes EvolvableModule can be created from a base module
        return EvolvableModule(base_module=module)
