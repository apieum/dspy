"""
This module contains specialized evaluators for the GEPA optimization process.
- ParentFastCompare: Quickly validates new candidates against their parents on a minibatch.
- FullTaskScores: Performs a comprehensive evaluation on a full, stable dataset.
These can be chained together to create a multi-phase evaluation pipeline.
"""

import logging
from typing import Callable, Optional, TYPE_CHECKING, List, Dict
import dspy
from dspy import Module
from .evaluator import Evaluator
from ..data.cohort import NewBorns, Survivors
from ..budget import Budget

if TYPE_CHECKING:
    pass  # No forward references needed

logger = logging.getLogger(__name__)


class ParentFastCompare(Evaluator):
    """
    An evaluator that performs a quick validation of new candidates (children)
    by comparing them against their parents on a small, random minibatch of data.

    This corresponds to Phase 1 of the original GEPA evaluation logic.
    """

    def __init__(self, metric: Callable, minibatch_size: int = 3, **kwargs):
        """
        Args:
            metric: The function to evaluate predictions against examples.
            minibatch_size: The number of examples to use for the validation minibatch.
        """
        super().__init__()
        self.metric = metric
        self.minibatch_size = minibatch_size
        self.split_strategy = None
        self.verbose = False

    def start_compilation(self, student: dspy.Module, split_strategy=None, verbose: bool=False) -> None:
        """Prepares the evaluator with the split strategy from Darwin."""
        self.split_strategy = split_strategy
        self.verbose = verbose

    def evaluate(self, new_borns: NewBorns, budget: Budget) -> Survivors:
        """
        Filters a cohort of new candidates, keeping only those that are "promising."

        A candidate is promising if:
        1. It has no parents (i.e., it's an initial candidate).
        2. It outperforms its parent on a random validation minibatch.
        """
        promising_candidates = []

        for candidate in new_borns.candidates:
            is_promising = False
            if not candidate.parents:
                # Initial candidates are always promising.
                is_promising = True
            else:
                is_improved, cost = self._validate_on_minibatch(candidate)
                budget.spend_on_evaluation(candidate.module, {"phase": "validation", "cost": cost})
                if is_improved:
                    is_promising = True

            if is_promising:
                promising_candidates.append(candidate)

        logger.info(f"ParentFastCompare: {len(promising_candidates)}/{len(new_borns.candidates)} candidates passed validation.")
        return Survivors(*promising_candidates, iteration=new_borns.iteration)

    def _validate_on_minibatch(self, child: 'Candidate') -> tuple[bool, int]:
        """Compares a child and parent on a random minibatch from the development set."""
        validation_minibatch = self._get_validation_minibatch(self.minibatch_size)
        if not validation_minibatch:
            return False, 0

        try:
            # DEBUG: Check actual instructions before validation
            from dspy.teleprompt.utils import get_signature
            child_predictors = child.module.predictors()
            if child_predictors:
                child_instruction = get_signature(child_predictors[0]).instructions
                logger.info(f"VALIDATION INSTRUCTION DEBUG: child instruction: {child_instruction}")

            for i, parent in enumerate(child.parents):
                parent_predictors = parent.module.predictors()
                if parent_predictors:
                    parent_instruction = get_signature(parent_predictors[0]).instructions
                    logger.info(f"VALIDATION INSTRUCTION DEBUG: parent {i} instruction: {parent_instruction}")

            # Phase 1: Validation on minibatch (n={len(validation_minibatch)} examples)
            logger.debug(f"Validation phase: evaluating on {len(validation_minibatch)} examples from internal validation set")
            child_scores = child.evaluate_on_batch(validation_minibatch, metric=self.metric, verbose=self.verbose)
            parent_avg_scores = []
            parent_scores = []
            for parent in child.parents:
                score = parent.evaluate_on_batch(validation_minibatch, metric=self.metric, verbose=self.verbose)
                parent_scores.append(score)
                parent_avg_scores.append(sum(score.values()) / len(score) if score else 0.0)

            avg_child = sum(child_scores.values()) / len(child_scores) if child_scores else 0
            avg_parent = min(parent_avg_scores) if parent_avg_scores else 0.0 # get the weakest parent score for comparison

            # Allow small tolerance for floating point precision and enable progression when equal
            tolerance = 0.01  # 1% tolerance
            is_improved = (avg_child >= avg_parent - tolerance)  # Accept equal or better scores
            cost = len(validation_minibatch) * len(child.parents)  # Cost for evaluating both child and parents

            logger.info(f"Validation result: parent={avg_parent:.3f}, child={avg_child:.3f}, passes_filter={is_improved}")
            logger.debug(f"Minibatch scores - child: {child_scores}, parents: {parent_scores}")
            logger.debug(f"Evaluated on tasks: {list(validation_minibatch.keys())}")
            return is_improved, cost

        except Exception as e:
            logger.warning(f"Minibatch validation failed: {e}")
            return False, len(validation_minibatch) * len(child.parents)

    def _get_validation_minibatch(self, size: int) -> Dict[int, dspy.Example]:
        """Get a minibatch using split strategy for validation."""
        if not self.split_strategy:
            return {}

        # Use split strategy to get evaluation minibatch (properly handles internal validation)
        selected = self.split_strategy.get_evaluation_minibatch(None, size)
        # Return as dict with task IDs as keys (for compatibility with existing code)
        return {i: example for i, example in enumerate(selected)}


class FullTaskScores(Evaluator):
    """
    An evaluator that computes scores for all candidates on the full, stable
    evaluation dataset.

    This corresponds to Phase 2 of the original GEPA evaluation logic. It assumes
    that the candidates it receives have already been validated as promising.
    """

    def __init__(self, metric: Callable, **kwargs):
        """
        Args:
            metric: The function to evaluate predictions against examples.
        """
        super().__init__()
        self.metric = metric
        self.split_strategy = None
        self.verbose = False

    def start_compilation(self, student: dspy.Module, split_strategy=None, verbose: bool=False) -> None:
        """Prepares the evaluator with the split strategy from Darwin."""
        self.split_strategy = split_strategy
        self.verbose = verbose

    def evaluate(self, new_borns: NewBorns, budget: Budget) -> Survivors:
        """
        Computes and assigns task scores for every candidate in the cohort
        on the full evaluation set.
        """
        if not self.split_strategy:
            raise ValueError("Split strategy not initialized.")

        # Phase 2: Comprehensive evaluation on full internal validation set
        evaluation_examples = self.split_strategy.internal_validation_set
        evaluation_set = {i: example for i, example in enumerate(evaluation_examples)}
        logger.debug(f"Full evaluation phase: assessing {len(new_borns.candidates)} candidates on {len(evaluation_set)} examples")

        for candidate in new_borns.candidates:
            # Comprehensive evaluation on complete internal validation set
            candidate.batch_task_scores(evaluation_set, metric=self.metric, verbose=self.verbose)

            # Report final candidate performance
            avg_score = candidate.average_task_score()
            logger.info(f"Comprehensive evaluation: candidate gen={candidate.generation_number} achieves μ={avg_score:.3f}")

            # Technical details for reproducibility
            predictors = candidate.module.predictors()
            if predictors:
                from dspy.teleprompt.utils import get_signature
                instruction = get_signature(predictors[0]).instructions
                module_id = id(candidate.module)
                predictor_id = id(predictors[0])
                logger.debug(f"Module ID={module_id}, Predictor ID={predictor_id}")
                logger.debug(f"Instruction: {instruction[:100]}...")

            budget.spend_on_evaluation(
                candidate.module,
                {"phase": "full_evaluation", "examples": len(evaluation_set)}
            )

        logger.info(f"Comprehensive evaluation completed for {len(new_borns.candidates)} candidates on {len(evaluation_set)} tasks")
        # All candidates that get a full evaluation are considered "survivors" of this stage.
        return Survivors(*new_borns.to_list(), iteration=new_borns.iteration)

GEPATwoPhasesEval = Evaluator.create_chain("GEPATwoPhasesEval", [ParentFastCompare, FullTaskScores])
