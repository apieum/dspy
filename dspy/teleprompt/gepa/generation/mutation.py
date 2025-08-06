"""Mutation-based candidate generation with reflective prompt mutation."""

import logging
from typing import List, Optional
import dspy
from dspy.teleprompt.utils import get_signature, set_signature

from .generator import Generator
from .reflective_mutation import ReflectiveMutation, PromptMutator
from ..data.candidate import Candidate
from ..evaluation.trace_collector import EnhancedTraceCollector
from ..evaluation.feedback import FeedbackResult
from ..data.cohort import Parents, NewBorns

logger = logging.getLogger(__name__)


class ModuleSelector:
    """Simple module selector for mutation."""

    def select_module(self, program: dspy.Module) -> int:
        """Select module index to mutate. For now, select first predictor."""
        predictors = program.predictors()
        return 0 if predictors else 0


class MutationGenerator(Generator):
    """Paper-compliant mutation generator using reflective prompt mutation.

    Implements the mutation strategy from GEPA paper, selecting a parent
    candidate and improving it through reflective prompt mutation based
    on feedback from generic feedback data.
    """

    def __init__(self,
                 prompt_mutator: Optional[PromptMutator] = None,
                 module_selector: Optional[ModuleSelector] = None,
                 feedback_collector: Optional[EnhancedTraceCollector] = None,
                 mutation_rate: float = 0.3,
                 population_size: int = 10):
        """Initialize mutation generator with required components.

        Args:
            prompt_mutator: Strategy for mutating prompts based on feedback
            module_selector: Strategy for selecting which module to mutate
            feedback_collector: Collector for gathering performance feedback
            mutation_rate: Probability of applying mutation
            population_size: Number of candidates to generate
        """
        self.prompt_mutator = prompt_mutator or ReflectiveMutation()
        self.module_selector = module_selector or ModuleSelector()
        self.feedback_collector = feedback_collector or EnhancedTraceCollector()
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        # This will be set during prepare_for_compilation()
        self.feedback_data: List[dspy.Example] = []

    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate new candidates through mutation of selected parents.

        Follows paper's Algorithm 1 approach:
        1. Use stochastic selection to sample 1 parent candidate from parents
        2. Collect feedback on parent using generic feedback_data
        3. Apply reflective prompt mutation
        4. Return single improved candidate
        """
        if parents.is_empty() or not self.feedback_data:
            return NewBorns()

        try:
            # Step 1: Stochastic selection of parent candidate (Algorithm 2 line 14)
            selected_parents = parents.sample_stochastic(1)
            if selected_parents.is_empty():
                return NewBorns()
                
            parent_candidate = selected_parents.first()

            # Step 2: Collect feedback on parent's performance
            feedback_result = self.feedback_collector.collect_feedback(
                parent_candidate.module, self.feedback_data, self._simple_metric
            )

            # Step 3: Apply reflective prompt mutation
            mutated_candidate = self._mutate_candidate(parent_candidate, feedback_result, parent_candidate.generation_number + 1, budget)

            if mutated_candidate is not None:
                return NewBorns(mutated_candidate)

        except Exception as e:
            logger.warning(f"Mutation failed: {e}")

        return NewBorns()


    def _mutate_candidate(self, parent_candidate: Candidate, feedback_result: FeedbackResult, iteration: int, budget=None) -> Optional[Candidate]:
        """Mutate candidate using reflective prompt mutation."""
        try:
            # Get predictors from parent module
            parent_module = parent_candidate.module
            predictors = parent_module.predictors()

            if not predictors:
                logger.warning("No predictors found in parent module")
                return None

            # Select module to mutate
            module_idx = self.module_selector.select_module(parent_module)
            if module_idx >= len(predictors):
                module_idx = 0

            predictor = predictors[module_idx]

            # Get current signature
            current_signature = get_signature(predictor)

            # Apply reflective mutation (LLM call happens here)
            new_signature = self.prompt_mutator.mutate_signature(current_signature, feedback_result)

            # Track budget for generation
            if budget is not None:
                budget.spend_on_generation(parent_candidate.module, {"type": "mutation", "iteration": iteration})

            # Create mutated module
            mutated_module = parent_module.deepcopy()
            mutated_predictors = mutated_module.predictors()

            if module_idx < len(mutated_predictors):
                set_signature(mutated_predictors[module_idx], new_signature)

            # Create new candidate
            new_candidate = Candidate(
                module=mutated_module,
                parents=[parent_candidate],
                generation_number=iteration
            )

            return new_candidate

        except Exception as e:
            logger.warning(f"Failed to mutate candidate: {e}")
            return None

    def _simple_metric(self, example: dspy.Example, prediction, trace=None) -> float:
        """Simple metric for feedback collection."""
        try:
            # Check if prediction has expected output field
            if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
                expected = str(example.answer).lower().strip()
                actual = str(prediction.answer).lower().strip()
                return 1.0 if expected == actual else 0.0

            # Fallback: check any string field
            for field_name in ['answer', 'output', 'result', 'response']:
                if hasattr(example, field_name) and hasattr(prediction, field_name):
                    expected = str(getattr(example, field_name)).lower().strip()
                    actual = str(getattr(prediction, field_name)).lower().strip()
                    return 1.0 if expected == actual else 0.0

            return 0.0
        except Exception:
            return 0.0

    def start_compilation(self, student: dspy.Module, training_data: List[dspy.Example]) -> None:
        """Prepare generator with training dataset when compilation begins.

        Uses training data as feedback data for reflective mutation.
        """
        self.feedback_data = training_data
