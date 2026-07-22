"""GEPA - Default evolutionary optimization strategy."""

import logging
import inspect
import random
import json
import importlib
import signal
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import dspy
from dspy.teleprompt.utils import get_signature, set_signature
from .base import BaseStrategy
from ..data.candidate import Candidate, example_id
from ..data.cohort import Cohort, NewBorns, Survivors, Parents
from ..result import Result, Success, Failure
from ..state import OptimizationCheckpoint
from ..generation import SingleMutationSampling, EpochShuffledBatchSampler
from ..evaluation import EvaluationCache
from ..evaluation import Metric

if TYPE_CHECKING:
    from ..config import DarwinConfig

logger = logging.getLogger(__name__)


def _tuple_tree(value):
    """Convert JSON-loaded RNG state lists back to nested tuples."""
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class GEPAStrategy(BaseStrategy[Result]):
    """Default implementation of evolutionary optimization strategy.

    Implements a simple evolutionary algorithm with the following steps:
    1. Initialize population with single candidate
    2. Evaluate candidates
    3. Select survivors
    4. Generate new candidates (mutation/crossover)
    5. Repeat until termination criteria met
    """

    def __init__(self, config: 'DarwinConfig'):
        super().__init__(config)
        self.algorithm_state = "initialize"  # initialize -> evaluate -> select -> generate -> repeat
        self.current_newborns: Optional[NewBorns] = None
        self.current_survivors: Optional[Survivors] = None
        self.current_parents: Optional[Parents] = None
        self._iteration_started = False
        self.history = []
        self.evaluation_cache = EvaluationCache()
        self._budget_exhaustion_notified = False
        self.rng = random.Random(config.seed)
        self.batch_sampler = None
        self._previous_signal_handlers = {}
        self._signal_stop_requested = False
        self._signal_stop_reason = None

    def start_compilation(
        self, student: dspy.Module, *, trainset: list[dspy.Example], devset: list[dspy.Example] | None = None, teacher: dspy.Module | None = None, **kwargs
    ) -> None:
        """Initialize the strategy with the compilation parameters."""
        self.student = student
        self.trainset = trainset
        self.devset = devset if devset is not None else []
        self.teacher = teacher
        # Components own compilation-scoped state. Recreate them for every
        # run so budgets, caches, selectors, and evaluators cannot leak across
        # repeated compilations of the same optimizer instance.
        self._budget = None
        self._selector = None
        self._generator = None
        self._crossover = None
        self._evaluator = None
        self.evaluation_cache = EvaluationCache()
        self.current_generation = 0
        self.best_candidate = None
        self.generations_without_improvement = 0
        self.history = []
        self._merge_due = False
        self._merge_attempts = 0
        self.rng = random.Random(self.config.seed)
        self.batch_sampler = self.config.batch_sampler or EpochShuffledBatchSampler(
            self.config.mutation_config.minibatch_size,
        )
        self._install_signal_handlers()

        # Centralize train/dev handling so experiments can inject a different
        # dataset policy without changing the strategy itself.
        manager_factory = self.config.dataset_manager_factory
        # Accept both the default factory class and an already-configured
        # factory instance.  The latter is useful when an experiment needs a
        # custom split policy or a stateful dataset manager.
        if isinstance(manager_factory, type):
            factory_parameters = inspect.signature(manager_factory).parameters
            factory_kwargs = {"split_ratio": self.config.validation_split}
            if "seed" in factory_parameters:
                factory_kwargs["seed"] = self.config.seed
            manager_factory = manager_factory(**factory_kwargs)
        # Official GEPA evaluates on trainset when no explicit validation set
        # is supplied.  Keep the dataset manager's standalone split behavior,
        # but make the strategy pass the reference semantics explicitly.
        validation_input = self.devset if self.devset else self.trainset
        manager = manager_factory.create(self.trainset, validation_input)
        self.dataset_manager = manager
        self.training_data = list(manager.get_eval_set().values())
        self.validation_data = list(
            manager.get_validation_minibatch(manager.num_dev_examples).values()
        )

        if self.config.verbose:
            logger.info(f"Data split: {len(self.training_data)} train, {len(self.validation_data)} validation, {len(self.devset)} test")

        self._notify("start_compilation", self.student, self.dataset_manager)

        if self.config.resume_from:
            if self._restore_checkpoint(self.config.resume_from, student):
                self._write_checkpoint()
                return

        self._write_checkpoint()

        # Initialize the first candidate
        initial_candidate = Candidate(self.student.deepcopy(), generation_number=0)
        self.current_newborns = NewBorns([initial_candidate], iteration=0)

        self.algorithm_state = "evaluate"  # Start with evaluation of the initial candidate

    def next_step(self) -> bool:
        """Implement the evolutionary algorithm state machine."""
        if self.should_terminate():
            return False

        if self.algorithm_state == "evaluate":
            self._evaluate_step()
        elif self.algorithm_state == "select":
            self._select_step()
        elif self.algorithm_state == "generate":
            self._generate_step()
        else:
            # Invalid state, terminate
            return False
        
        return True

    def terminate_compilation(self) -> Result:
        """Get the result of the optimization process."""
        final_candidate = self._select_final_candidate()
        self.best_candidate = final_candidate
        if self.best_candidate:
            all_candidates = set()
            all_candidates.update(getattr(self.selector, "task_wins", {}).keys())
            for winners in getattr(self.selector, "example_best_candidates", {}).values():
                all_candidates.update(winners)
            if not all_candidates:
                all_candidates.add(self.best_candidate)
            ordered_candidates = [self.best_candidate] + [
                candidate for candidate in all_candidates
                if candidate is not self.best_candidate
            ]
            parents = {
                candidate: list(candidate.parents)
                for candidate in ordered_candidates
            }
            val_subscores = {
                candidate: {
                    score.id: float(score.value)
                    for score in candidate.scores
                }
                for candidate in ordered_candidates
            }
            winners = {
                task_id: set(candidates)
                for task_id, candidates in getattr(
                    self.selector, "example_best_candidates", {}
                ).items()
            }
            result = Success(
                candidates=ordered_candidates,
                history=list(self.history),
                best_candidate=self.best_candidate,
                parents=parents,
                val_subscores=val_subscores,
                per_val_instance_best_candidates=winners,
            )
        else:
            result = Failure(
                reason="Compilation failed, no candidate was found."
            )
        result_module = self.best_candidate.module if self.best_candidate is not None else self.student
        if self.best_candidate is not None:
            result_module._compiled = True
        self._notify("finish_compilation", result_module)
        self._write_checkpoint(completed=True)
        self._restore_signal_handlers()
        return result

    def _install_signal_handlers(self) -> None:
        if not self.config.handle_signals:
            return
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
        except (ValueError, OSError):
            # Signal handlers can only be installed by the main thread and
            # are unavailable on some platforms. Optimization still works.
            self._previous_signal_handlers.clear()

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass
        self._previous_signal_handlers.clear()

    def _handle_signal(self, signum, frame) -> None:
        self._signal_stop_requested = True
        self._signal_stop_reason = signal.Signals(signum).name
        try:
            self._write_checkpoint(completed=False)
        except Exception:
            logger.exception("Unable to write Darwin checkpoint after %s", self._signal_stop_reason)

    def get_checkpoint(self, completed: bool = False) -> OptimizationCheckpoint:
        """Return a JSON-safe snapshot of the current optimization state."""
        candidates = []
        tracked = set()
        for attribute, cohort in (
            (name, value)
            for name, value in self.__dict__.items()
            if isinstance(value, Cohort)
        ):
            if cohort is None:
                continue
            for candidate in cohort:
                if id(candidate) in tracked:
                    continue
                tracked.add(id(candidate))
                candidates.append({
                    "id": id(candidate),
                    "cohort_attributes": [attribute],
                    "cohort_type": (
                        f"{type(cohort).__module__}:{type(cohort).__qualname__}"
                        if cohort is not None else None
                    ),
                    "generation": candidate.generation_number,
                    "score": candidate.average_score(),
                    "parents": [id(parent) for parent in candidate.parents],
                    "scores": [
                        {
                            "id": score.id,
                            "value": float(score.value),
                            "feedback": score.feedback,
                            "objective_scores": dict(score.objective_scores),
                        }
                        for score in candidate.scores
                    ],
                    "instructions": [
                        getattr(get_signature(predictor), "instructions", "")
                        for predictor in candidate.module.predictors()
                    ],
                    "creation_metadata": self._serialize_creation_metadata(
                        candidate.creation_metadata
                    ),
                    "proposal_minibatch_ids": [
                        example_id(example)
                        for example in (candidate.proposal_minibatch or [])
                    ],
                })
                # A candidate can exist in multiple active cohorts. Preserve
                # all roles without duplicating its serialized record.
                if any(item["id"] == id(candidate) for item in candidates[:-1]):
                    existing = next(item for item in candidates if item["id"] == id(candidate))
                    existing.setdefault("cohort_attributes", []).append(attribute)
                    candidates.pop()
        remaining = self.budget.get_remaining()
        return OptimizationCheckpoint(
            generation=self.current_generation,
            algorithm_state=self.algorithm_state,
            history=list(self.history),
            budget=remaining if isinstance(remaining, dict) else {"remaining": remaining},
            candidates=candidates,
            strategy_state=self._get_checkpoint_strategy_state(),
            rng_state=self.rng.getstate(),
            stop_reason=self._signal_stop_reason,
            completed=completed,
        )

    def _get_checkpoint_strategy_state(self) -> dict:
        """Return JSON-safe scalar state needed for deterministic resumption."""
        return {
            "generations_without_improvement": getattr(self, "generations_without_improvement", 0),
            "merge_due": getattr(self, "_merge_due", False),
            "merge_attempts": getattr(self, "_merge_attempts", 0),
            "iteration_started": getattr(self, "_iteration_started", False),
            "budget_exhaustion_notified": getattr(self, "_budget_exhaustion_notified", False),
        }

    @staticmethod
    def _serialize_creation_metadata(metadata):
        """Keep metadata JSON-safe while preserving candidate references."""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, Candidate):
                serialized[f"{key}_id"] = id(value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                serialized[str(key)] = value
        return serialized

    def _restore_checkpoint(self, path: str, student: dspy.Module) -> bool:
        """Restore a compilation from a Darwin checkpoint manifest."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Darwin checkpoint not found: {path}")
        checkpoint = OptimizationCheckpoint.from_dict(
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
        )

        records = checkpoint.candidates
        restored = {}
        for record in records:
            module = student.deepcopy()
            predictors = module.predictors()
            for predictor, instruction in zip(predictors, record.get("instructions", [])):
                set_signature(
                    predictor,
                    get_signature(predictor).with_instructions(instruction),
                )
            candidate = Candidate(
                module,
                generation_number=int(record.get("generation", 0)),
                creation_metadata=record.get("creation_metadata", {}),
            )
            candidate.scores = [
                Metric(
                    score.get("value", 0.0),
                    id=str(score.get("id", "")),
                    feedback=score.get("feedback", ""),
                    objective_scores=score.get("objective_scores", {}),
                )
                for score in record.get("scores", [])
            ]
            example_by_id = {
                example_id(example): example
                for example in self.training_data + self.validation_data
            }
            candidate.proposal_minibatch = [
                example_by_id[task_id]
                for task_id in record.get("proposal_minibatch_ids", [])
                if task_id in example_by_id
            ] or None
            restored[record.get("id")] = candidate

        for record in records:
            candidate = restored[record.get("id")]
            candidate.parents = [
                restored[parent_id]
                for parent_id in record.get("parents", [])
                if parent_id in restored
            ]
            ancestor_id = candidate.creation_metadata.pop("ancestor_candidate_id", None)
            if ancestor_id in restored:
                candidate.creation_metadata["ancestor_candidate"] = restored[ancestor_id]

        self.history = list(checkpoint.history)
        self.current_generation = checkpoint.generation
        self.algorithm_state = "terminate" if checkpoint.completed else checkpoint.algorithm_state
        strategy_state = checkpoint.strategy_state or {}
        self.generations_without_improvement = int(
            strategy_state.get("generations_without_improvement", 0)
        )
        self._merge_due = bool(strategy_state.get("merge_due", False))
        self._merge_attempts = int(strategy_state.get("merge_attempts", 0))
        self._iteration_started = bool(strategy_state.get("iteration_started", False))
        self._budget_exhaustion_notified = bool(
            strategy_state.get("budget_exhaustion_notified", False)
        )
        if checkpoint.rng_state is not None:
            self.rng.setstate(_tuple_tree(checkpoint.rng_state))

        self._budget = None
        budget = self.budget
        remaining = checkpoint.budget
        if hasattr(budget, "max_calls") and "calls" in remaining:
            budget.consumed_calls = max(0, budget.max_calls - int(remaining["calls"]))
            if hasattr(budget, "evaluation_max_calls"):
                budget.evaluation_calls = max(
                    0, budget.evaluation_max_calls - int(remaining.get("evaluation_calls", budget.evaluation_calls))
                )
            if hasattr(budget, "generation_max_calls"):
                budget.generation_calls = max(
                    0, budget.generation_max_calls - int(remaining.get("generation_calls", budget.generation_calls))
                )

        cohorts = {}
        for record in records:
            for attribute in record.get("cohort_attributes", []):
                cohorts.setdefault(attribute, []).append(record)

        def restore_cohort(attribute, records_for_cohort):
            cohort_type = next(
                (record.get("cohort_type") for record in records_for_cohort
                 if record.get("cohort_type")),
                "dspy.teleprompt.darwin.data.cohort:Cohort",
            )
            module_name, qualname = cohort_type.split(":", 1)
            cohort_cls = importlib.import_module(module_name)
            for part in qualname.split("."):
                cohort_cls = getattr(cohort_cls, part)
            candidates_for_role = [
                restored[record.get("id")]
                for record in records_for_cohort
            ]
            return cohort_cls(*candidates_for_role, iteration=self.current_generation)

        # Restore every serialized cohort attribute without knowing which
        # algorithm owns it or what its state-machine vocabulary is.
        for attribute, records_for_cohort in cohorts.items():
            setattr(self, attribute, restore_cohort(attribute, records_for_cohort))

        # Rebuild the selector's per-task Pareto state from restored scores.
        self._selector = self.config.selection()
        configure = getattr(self._selector, "configure", None)
        if callable(configure):
            configure(self.config)
        selector_cohort = next(iter(cohorts.values()), None)
        if selector_cohort:
            selector_type = next(
                record.get("cohort_type") for record in selector_cohort
                if record.get("cohort_type")
            )
            module_name, qualname = selector_type.split(":", 1)
            selector_cohort_cls = importlib.import_module(module_name)
            for part in qualname.split("."):
                selector_cohort_cls = getattr(selector_cohort_cls, part)
        else:
            selector_cohort_cls = Cohort
        self._selector.update_scores_batch(
            selector_cohort_cls(*restored.values(), iteration=self.current_generation)
        )
        self.best_candidate = max(
            restored.values(), key=lambda candidate: candidate.average_score(), default=None
        )
        return True

    def _write_checkpoint(self, completed: bool = False) -> None:
        if not self.config.checkpoint_path:
            return
        path = Path(self.config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.get_checkpoint(completed=completed).to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _select_final_candidate(self) -> Optional[Candidate]:
        """Select the best aggregate candidate after optimization.

        ``candidate_selection_strategy`` belongs to proposal-parent sampling.
        GEPA's final result is the best generalist, so exploration strategies
        must not randomly replace it at termination.
        """
        candidate = self.best_candidate
        selector_best = getattr(self.selector, "best_candidate", None)
        if callable(selector_best):
            try:
                selected = selector_best()
                if (
                    isinstance(selected, Candidate)
                    and (
                        candidate is None
                        or selected.average_score() >= candidate.average_score()
                    )
                ):
                    candidate = selected
            except (RuntimeError, ValueError):
                # A selector may have no accumulated scores when compilation
                # ends before the first evaluation completes.
                pass
        return candidate

    def _evaluate_step(self):
        """Evaluate current candidates."""
        if self.config.verbose:
            logger.info(f"Evaluating candidates in generation {self.current_generation}")

        if self.current_newborns is None or self.current_newborns.is_empty():
            # No candidates to evaluate, move to selection
            self.algorithm_state = "select"
            return

        # Evaluate the newborns (including initial candidate)
        self.current_survivors = self.evaluator.evaluate(self.current_newborns, self.budget)

        survivors = set(self.current_survivors.candidates)
        for candidate in self.current_newborns:
            self._notify("candidate_evaluated", candidate, candidate in survivors)

        generation_best_score = None

        # Update best candidate tracking
        if self.current_survivors and not self.current_survivors.is_empty():
            best_in_generation = max(self.current_survivors.candidates,
                                   key=lambda c: c.average_score() if c.average_score() is not None else -1)
            generation_best_score = best_in_generation.average_score()

            if (self.best_candidate is None or
                (best_in_generation.average_score() is not None and self.best_candidate.average_score() is not None and
                 best_in_generation.average_score() > self.best_candidate.average_score())):
                self.best_candidate = best_in_generation
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

        self.history.append({
            "generation": self.current_generation,
            "evaluated_candidates": len(self.current_newborns),
            "surviving_candidates": len(self.current_survivors),
            "best_score": generation_best_score,
        })
        self._write_checkpoint()

        self.algorithm_state = "select"

    def _select_step(self):
        """Select candidates for next generation."""
        if self.config.verbose:
            logger.info(f"Selecting candidates for generation {self.current_generation + 1}")

        if self.current_survivors is None or self.current_survivors.is_empty():
            # No survivors to select from, move to generate
            self.algorithm_state = "generate"
            return

        if self.config.use_merge and self._merge_attempts < self.config.max_merge_invocations:
            # Official GEPA schedules a merge opportunity only after a
            # reflective mutation was accepted. The seed candidate and merge
            # children do not schedule another merge opportunity.
            self._merge_due = any(
                candidate.parents
                and candidate.creation_metadata.get("merge_type") != "system_aware"
                for candidate in self.current_survivors
            )

        # Promote survivors to parents for next generation
        self.current_parents = self.selector.promote(self.current_survivors)
        if self._iteration_started:
            self._notify("finish_iteration", self.current_generation, self.current_parents, self.budget)
            self._iteration_started = False
        self.algorithm_state = "generate"
        self._write_checkpoint()

    def _generate_step(self):
        """Generate new candidates."""
        self.current_generation += 1

        if self.config.verbose:
            logger.info(f"Generating candidates for generation {self.current_generation}")

        if self.current_parents is None or self.current_parents.is_empty():
            # No parents available, terminate by setting state that leads to exit
            self.algorithm_state = "terminate"
            return

        self._notify("start_iteration", self.current_generation, self.current_parents, self.budget)
        self._iteration_started = True

        # Official GEPA gives a merge one opportunity after a successful
        # mutation, then falls back to reflective mutation if no merge is
        # possible. This keeps merging opt-in while preserving the separate
        # GEPAAdaptive strategy for merge-anytime experiments.
        self.current_newborns = None
        if self._merge_due:
            self._merge_due = False
            self._merge_attempts += 1
            merged = self.crossover_generator.generate(self.current_parents, self.budget)
            if not merged.is_empty():
                self.current_newborns = merged

        if self.current_newborns is None:
            self.current_newborns = self.generator.generate_batch(
                self.current_parents,
                self.config.proposals_per_generation,
                self.budget,
                sampling_strategy=self.config.sampling_strategy or SingleMutationSampling(),
                batch_sampler=self.batch_sampler,
                rng=self.rng,
            )

        if self.current_newborns.is_empty() and self.budget <= 0:
            self._notify("budget_exhausted", self.budget)

        # Cycle back to evaluation
        self.algorithm_state = "evaluate"
        self._write_checkpoint()
