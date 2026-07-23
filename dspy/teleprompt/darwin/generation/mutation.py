"""ReflectivePromptMutation - Evolutionary mutation using reflection on feedback."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Dict
import random

import dspy
from .generator import Generator
from .reflection_strategy import ReflectionStrategy
from .evolvable_module import EvolvableModule
from .prompt_mutator import ReflectivePromptMutator
from .dspy_utils import get_predictors
from .config import ReflectiveMutationConfig, ModuleSelectionStrategy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DarwinConfig
from ..data.candidate import Candidate
from ..data.cohort import Parents, NewBorns
from ..budget import BudgetEvent, BudgetExhaustedError

logger = logging.getLogger(__name__)


class ReflectivePromptMutation(Generator):
    """
    GEPA's reflective mutation, focused purely on generation.
    """

    def __init__(self,
                 *,
                 config: DarwinConfig,
                 feedback_provider=None,
                 feedback_data: List[dspy.Example] = None,
                 reflection_strategy: Optional[ReflectionStrategy] = None,
                 reflection_lm: Optional[Any] = None,
                 module_selection: str = "round_robin",
                 candidate_selection_strategy: Any = "pareto",
                 max_retries: int = 1,
                 assessor=None,
                 rng=None):
        super().__init__()
        self.config = config
        mutation_config = config.mutation_config
        if feedback_provider is None and assessor is not None:
            from .feedback import FeedbackProvider
            feedback_provider = FeedbackProvider(
                assessor=assessor,
                failure_score=config.failure_score,
            )
        feedback_provider = mutation_config.feedback_provider or feedback_provider
        reflection_strategy = mutation_config.reflection_strategy or reflection_strategy
        module_selection = mutation_config.module_selection_strategy.value
        max_retries = mutation_config.max_retries
        reflection_lm = config.reflection_lm or reflection_lm
        candidate_selection_strategy = config.candidate_selection_strategy

        if feedback_provider is None:
            raise ValueError("ReflectivePromptMutation requires a FeedbackProvider")

        self.feedback_provider = feedback_provider
        self.feedback_data = feedback_data or []
        self.minibatch_size = mutation_config.minibatch_size
        self.reflection_strategy = reflection_strategy or mutation_config.reflection_strategy
        self.reflection_lm = reflection_lm
        self.reuse_parent_rollouts = (
            config.reuse_parent_rollouts
        )
        self.module_selection = module_selection
        self.candidate_selection_strategy = candidate_selection_strategy
        self.max_retries = max(1, max_retries)
        self.use_abstract_feedback = mutation_config.use_abstract_feedback
        self.perfect_score = config.perfect_score
        # Direct generator users historically expect a proposal even for a
        # perfect toy metric; strategy-owned GEPA runs use the reference skip.
        self.skip_perfect_score = (
            config.skip_perfect_score
        )

        self.next_module_idx = 0
        self.rng = rng or random.Random()
        self._next_module_by_parent = {}

    def start_compilation(
        self,
        student: dspy.Module,
        dataset_manager=None,
        *,
        feedback_data: Optional[List[dspy.Example]] = None,
        verbose: bool = False,
    ) -> None:
        """Set verbose mode for the generator."""
        if feedback_data is not None:
            self.feedback_data = feedback_data
        elif dataset_manager is not None:
            self.feedback_data = list(
                dataset_manager.get_feedback_minibatch(
                    self.minibatch_size
                ).values()
            )
        self.dataset_manager = dataset_manager
        self.next_module_idx = 0
        self._next_module_by_parent = {}
        self.verbose = verbose

    def generate(self, parents: Parents, budget=None) -> NewBorns:
        """Generate a new candidate without validation."""
        if parents.is_empty() or not self.feedback_data:
            return NewBorns()

        try:
            parent = self._select_parent(parents)
            if parent is None:
                return NewBorns()

            predictors = get_predictors(parent.module)
            if not predictors:
                return NewBorns()

            minibatch = self._get_feedback_minibatch()
            if not minibatch:
                return NewBorns()

            # Feedback execution evaluates the parent once per example and
            # reflection consumes one additional LM call. Reserve the whole
            # generation before making any request so max_calls is a hard
            # upper bound on Darwin's expected LM work.
            mutation_count = (
                len(predictors)
                if self.module_selection == ModuleSelectionStrategy.ALL.value
                else 1
            )
            generation_cost = len(minibatch) + mutation_count
            last_error = None
            for _ in range(self.max_retries):
                try:
                    module_idx = self._select_target_module(len(predictors), parent=parent)
                    evolvable = self._ensure_evolvable(parent.module)
                    # A local setup failure is not billable. Reserve the
                    # attempt only when an LM operation can actually start.
                    if budget is not None:
                        budget.spend(
                            BudgetEvent(
                                "generation",
                                generation_cost,
                                {"type": "reflective_mutation"},
                            )
                        )
                    if self.module_selection in {
                        ModuleSelectionStrategy.ALL.value,
                        ModuleSelectionStrategy.FAILED_ONLY.value,
                    }:
                        feedback_targets = list(range(len(predictors)))
                    else:
                        feedback_targets = [module_idx]

                    # Capture one parent rollout, then ask for predictor-level
                    # feedback from that shared trace.  This keeps ALL and
                    # FAILED_ONLY from reusing predictor-0 diagnostics while
                    # avoiding one extra program execution per predictor.
                    feedback_by_module = evolvable.collect_traces_and_evaluate_many(
                        minibatch,
                        self.feedback_provider,
                        feedback_targets,
                    )
                    if self.module_selection == ModuleSelectionStrategy.FAILED_ONLY.value:
                        trace_feedback = feedback_by_module[feedback_targets[0]]
                        target_modules = [self._select_failed_module(
                            trace_feedback.traces, len(predictors), parent
                        )]
                    elif self.module_selection == ModuleSelectionStrategy.ALL.value:
                        target_modules = feedback_targets
                    else:
                        target_modules = [module_idx]
                    feedback = feedback_by_module[target_modules[0]]

                    # Reuse the parent rollout for acceptance when an
                    # evaluation cache is attached by the strategy.  This is
                    # the reference GEPA behavior: one parent execution feeds
                    # both reflection and parent/child comparison.
                    evaluation_cache = getattr(self, "evaluation_cache", None)
                    if evaluation_cache is not None and self.reuse_parent_rollouts:
                        for example, metric in zip(feedback.examples, feedback.metrics):
                            evaluation_cache.put(parent, example, metric)

                    if (
                        self.skip_perfect_score
                        and self.perfect_score is not None
                        and feedback.scores
                        and all(float(score) >= self.perfect_score for score in feedback.scores)
                    ):
                        return NewBorns(iteration=parents.iteration)

                    # Evolve using a PromptMutator strategy
                    mutator = ReflectivePromptMutator(
                        self.reflection_strategy,
                        self.reflection_lm,
                        use_abstract_feedback=self.use_abstract_feedback,
                    )
                    child_module = evolvable
                    for target_module in target_modules:
                        target_feedback = feedback_by_module[target_module]
                        child_module = mutator.mutate(
                            child_module,
                            target_feedback,
                            target_module,
                            verbose=getattr(self, "verbose", False),
                        )

                    child_candidate = Candidate(
                        module=child_module,
                        generation_number=parent.generation_number + 1,
                        parents=[parent],
                        proposal_minibatch=list(minibatch.values()),
                    )

                    return NewBorns(child_candidate, iteration=parents.iteration)
                except Exception as e:
                    last_error = e

            if last_error is not None:
                raise last_error
            return NewBorns()

        except Exception as e:
            self.publish('mutation_failure', None, {'reason': f'Reflective prompt mutation failed: {e}'})
            return NewBorns()

    def _select_parent(self, parents: Parents) -> Optional[Candidate]:
        """Select the parent used for the next proposal.

        GEPA's candidate selector controls every proposal task.  Keeping this
        decision in the generator lets Darwin retain its component-oriented
        architecture while making the configured strategy effective during
        search, not only when choosing the final result.
        """
        if parents.is_empty():
            return None

        selector = self.candidate_selection_strategy
        if callable(selector) and not isinstance(selector, str):
            selected = selector(parents, self.rng)
            if isinstance(selected, Candidate):
                return selected
            if hasattr(selected, "first") and not selected.is_empty():
                return selected.first()
            raise TypeError("candidate selector must return a Candidate or non-empty cohort")

        candidates = parents.to_list()
        if selector == "current_best":
            return max(candidates, key=lambda candidate: candidate.total_score())

        if selector == "epsilon_greedy":
            if self.rng.random() < 0.1:
                return self.rng.choice(candidates)
            return max(candidates, key=lambda candidate: candidate.total_score())

        if selector == "top_k_pareto":
            top_candidates = sorted(
                candidates,
                key=lambda candidate: candidate.total_score(),
                reverse=True,
            )[:5]
            if len(top_candidates) == 1:
                return top_candidates[0]
            top_cohort = parents.__class__(
                *top_candidates,
                iteration=parents.iteration,
                **parents._new_weights(top_candidates),
            )
            return top_cohort.sample_stochastic(1, rng=self.rng).first()

        if selector == "pareto":
            return parents.sample_stochastic(1, rng=self.rng).first()

        raise ValueError(
            "candidate_selection_strategy must be one of: pareto, current_best, "
            "epsilon_greedy, top_k_pareto, or a callable"
        )

    def _select_target_module(self, num_modules: int, parent: Optional[Candidate] = None) -> int:
        """Select module to mutate."""
        if self.module_selection == ModuleSelectionStrategy.FAILED_ONLY.value:
            # The trace is needed to identify the failed predictor; generate()
            # refines this provisional target after the single parent rollout.
            return 0
        if self.module_selection == ModuleSelectionStrategy.ROUND_ROBIN.value:
            key = id(parent) if parent is not None else None
            module_idx = self._next_module_by_parent.get(key, 0) % num_modules
            self._next_module_by_parent[key] = module_idx + 1
            return module_idx
        elif self.module_selection == ModuleSelectionStrategy.RANDOM.value:
            return self.rng.randint(0, num_modules - 1)
        elif self.module_selection == ModuleSelectionStrategy.ALL.value:
            return 0
        elif self.module_selection == ModuleSelectionStrategy.WORST_PERFORMING.value:
            return self._select_worst_performing_module(parent, num_modules)
        else:
            raise ValueError(f"Unknown module selection strategy: {self.module_selection}")

    def _select_failed_module(self, traces, num_modules: int, parent=None) -> int:
        """Choose the predictor whose execution trace contains a failure."""
        failures = [0] * num_modules
        for trace in traces or []:
            for index, entry in enumerate(trace or []):
                if index >= num_modules or len(entry) < 3:
                    continue
                _, inputs, outputs = entry[:3]
                values = list((inputs or {}).values()) + list((outputs or {}).values())
                text = " ".join(str(value) for value in values).lower()
                if (
                    "failedprediction" in text
                    or "error" in text
                    or "exception" in text
                    or any("failedprediction" in type(value).__name__.lower() for value in values)
                ):
                    failures[index] += 1
        if any(failures):
            return max(range(num_modules), key=lambda index: failures[index])
        return self._select_target_module_without_failed_only(num_modules, parent)

    def _select_target_module_without_failed_only(self, num_modules: int, parent=None) -> int:
        key = id(parent) if parent is not None else None
        module_idx = self._next_module_by_parent.get(key, 0) % num_modules
        self._next_module_by_parent[key] = module_idx + 1
        return module_idx

    def _select_worst_performing_module(self, parent: Candidate, num_modules: int) -> int:
        if parent is None or not parent.scores:
            key = id(parent) if parent is not None else None
            module_idx = self._next_module_by_parent.get(key, 0) % num_modules
            self._next_module_by_parent[key] = module_idx + 1
            return module_idx

        module_errors = [0.0] * num_modules
        for score in parent.scores:
            trace = getattr(score, "trace", None) or {}
            module_idx = trace.get("module_idx")
            if isinstance(module_idx, int) and 0 <= module_idx < num_modules:
                module_errors[module_idx] += 1.0 - float(score.value)

        if sum(module_errors) == 0:
            key = id(parent) if parent is not None else None
            module_idx = self._next_module_by_parent.get(key, 0) % num_modules
            self._next_module_by_parent[key] = module_idx + 1
            return module_idx
        return max(range(num_modules), key=lambda idx: module_errors[idx])


    def _get_feedback_minibatch(self) -> Dict[int, dspy.Example]:
        """Get the feedback data provided by strategy."""
        feedback_data = getattr(self, "_active_feedback_data", None)
        if feedback_data is None:
            feedback_data = self.feedback_data
        if not feedback_data:
            return {}

        return {i: example for i, example in enumerate(feedback_data)}

    def _ensure_evolvable(self, module: dspy.Module) -> EvolvableModule:
        """Wrap DSPy module as EvolvableModule."""
        if isinstance(module, EvolvableModule):
            return module

        # This assumes EvolvableModule can be created from a base module
        return EvolvableModule(base_module=module)
