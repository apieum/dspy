"""GEPA - Default evolutionary optimization strategy."""

import inspect
import random
import json
import importlib
import signal
import os
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import dspy
from dspy.teleprompt.utils import get_signature, set_signature
from .base import BaseStrategy
from ..workflow import Workflow
from ..data.candidate import Candidate, example_id
from ..data.cohort import Cohort, NewBorns, Survivors, Parents
from ..result import Result, Success, Failure
from ..state import OptimizationCheckpoint
from ..evaluation import EvaluationCache
from ..evaluation import Metric

if TYPE_CHECKING:
    from ..config import GEPAConfig

def _tuple_tree(value):
    """Convert JSON-loaded RNG state lists back to nested tuples."""
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class GEPAWorkflow(Workflow[Result]):
    """GEPA's concrete workflow and its algorithm-specific runtime state.

    Implements a simple evolutionary algorithm with the following steps:
    1. Initialize population with single candidate
    2. Evaluate candidates
    3. Select survivors
    4. Generate new candidates (mutation/crossover)
    5. Repeat until termination criteria met
    """

    def __init__(self, config: 'GEPAConfig', *, notify=None):
        super().__init__(config, notify=notify)
        # Compilation-scoped runtime fields live in the workflow.  Keeping
        # them initialized here also makes the component facade usable before
        # ``start_compilation`` (for inspection and unit-test composition).
        self.current_generation = 0
        self.best_candidate = None
        self.generations_without_improvement = 0
        self.trainset = []
        self.devset = []
        self.training_data = []
        self.validation_data = []
        self.dataset_manager = None
        self.student = None
        self.teacher = None
        self._budget = None
        self._selector = None
        self._generator = None
        self._crossover = None
        self._evaluator = None
        self._merge_due = False
        self._merge_attempts = 0
        self._budget_exhaustion_notified = False
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
        self._phase_actions = {
            "initialize": self.initialize,
            "evaluate": self.evaluate,
            "select": self.select,
            "generate": self.generate,
        }

    def _create_minibatch(self, data: List[dspy.Example], size: int) -> List[dspy.Example]:
        """Sample a deterministic minibatch for the current GEPA run."""
        if not data:
            return []
        if len(data) <= size:
            return data.copy()
        return self.rng.sample(data, size)

    @property
    def budget(self):
        if not hasattr(self, "_budget") or self._budget is None:
            self._budget = self.config.budget(config=self.config)
        return self._budget

    @property
    def selector(self):
        if not hasattr(self, "_selector") or self._selector is None:
            self._selector = self.config.selection(config=self.config)
        return self._selector

    @property
    def generator(self):
        if not hasattr(self, "_generator") or self._generator is None:
            self._generator = self._instantiate_generator(self.config.mutation)
        return self._generator

    @property
    def crossover_generator(self):
        if not hasattr(self, "_crossover") or self._crossover is None:
            if self.config.crossover is None:
                raise RuntimeError("No crossover generator is configured")
            self._crossover = self._instantiate_generator(self.config.crossover)
        return self._crossover

    @property
    def evaluator(self):
        if not hasattr(self, "_evaluator") or self._evaluator is None:
            minibatch_data = self._create_minibatch(
                self.validation_data, self.config.minibatch_size
            )
            self._evaluator = self.config.evaluation(
                config=self.config,
                minibatch_data=minibatch_data,
                validation_data=self.validation_data,
                evaluation_cache=self.evaluation_cache,
            )
            self._evaluator.start_compilation(
                getattr(self, "student", None),
                dataset_manager=self.dataset_manager,
                verbose=self.config.verbose,
            )
        return self._evaluator

    def should_terminate(self) -> bool:
        """Apply GEPA's stopping policy before dispatching the next phase."""
        if getattr(self, "_signal_stop_requested", False):
            return True
        for stopper in self.config.stoppers:
            if stopper(self):
                return True
        if self.budget <= 0:
            if not getattr(self, "_budget_exhaustion_notified", False):
                self.notify("budget_exhausted", self.budget)
                self._budget_exhaustion_notified = True
            return True

        remaining = self.budget.get_remaining()
        if isinstance(remaining, dict):
            if (
                self.algorithm_state == "generate"
                and "generation_calls" in remaining
                and remaining["generation_calls"] <= 0
            ):
                return True
            if (
                self.algorithm_state in {"select", "generate"}
                and "evaluation_calls" in remaining
                and remaining["evaluation_calls"] <= 0
            ):
                return True
        if (
            self.config.patience is not None
            and self.generations_without_improvement >= self.config.patience
        ):
            return True
        if (
            self.algorithm_state == "generate"
            and self.config.max_iterations is not None
            and self.current_generation >= self.config.max_iterations
        ):
            return True
        return False

    def _instantiate_generator(self, generator_factory):
        from ..generation.feedback import FeedbackProvider

        mutation_config = self.config.mutation_config
        feedback_assessor = (
            mutation_config.feedback_provider.assessor
            if mutation_config and mutation_config.feedback_provider is not None
            else self.config.enhanced_feedback or self.config.fitness_function
        )
        feedback_function = mutation_config.enhanced_feedback_function if mutation_config else None
        feedback_provider = (
            mutation_config.feedback_provider
            if mutation_config and mutation_config.feedback_provider is not None
            else FeedbackProvider(
                assessor=feedback_assessor,
                feedback_function=feedback_function,
                failure_score=self.config.failure_score,
            )
        )
        feedback_data = self._create_minibatch(
            self.training_data,
            mutation_config.minibatch_size
            if mutation_config
            else self.config.minibatch_size,
        )
        generator = generator_factory(
            feedback_provider=feedback_provider,
            feedback_data=feedback_data,
            assessor=feedback_assessor,
            config=self.config,
            rng=self.rng,
        )
        generator.feedback_pool = list(self.training_data)
        generator.evaluation_cache = self.evaluation_cache
        student = getattr(self, "student", None)
        if student is not None:
            generator.start_compilation(
                student,
                dataset_manager=self.dataset_manager,
                feedback_data=feedback_data,
                verbose=self.config.verbose,
            )
        return generator

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
        self.current_newborns = None
        self.current_survivors = None
        self.current_parents = None
        self.algorithm_state = "initialize"
        self.rng = random.Random(self.config.seed)
        self.batch_sampler = self.config.batch_sampler
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
            self.notify(
                "log",
                "info",
                f"Data split: {len(self.training_data)} train, "
                f"{len(self.validation_data)} validation, {len(self.devset)} test",
            )

        if self.config.resume_from:
            if self._restore_checkpoint(self.config.resume_from, student):
                self._write_checkpoint()
                return

        self._write_checkpoint()

        # The initial cohort is built by the first graph phase.  This keeps
        # initialization polymorphic for alternate strategies and graphs.
        self.algorithm_state = "initialize"

    def next_step(self) -> bool:
        """Execute the action registered for the current GEPA phase."""
        if self.should_terminate():
            return False
        try:
            action = self._phase_actions[self.algorithm_state]
        except KeyError as exc:
            raise RuntimeError(f"Unknown GEPA phase: {self.algorithm_state!r}") from exc
        action()
        return True

    def finish_compilation(self) -> Result:
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
        if self.best_candidate is not None:
            self.best_candidate.module._compiled = True
        self._write_checkpoint(completed=True)
        self._restore_signal_handlers()
        return result

    def initialize(self) -> None:
        """Create the seed cohort for the current compilation."""
        initial_candidate = Candidate(self.student.deepcopy(), generation_number=0)
        self.current_newborns = NewBorns([initial_candidate], iteration=0)
        self.algorithm_state = "evaluate"

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
            self.notify(
                "log",
                "exception",
                "Unable to write Darwin checkpoint after %s",
                self._signal_stop_reason,
                exc_info=True,
            )

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
            # Candidate scores are valid cache entries for the corresponding
            # examples.  Seeding the compilation cache prevents a resumed
            # run from spending new LM calls merely to re-evaluate restored
            # parents.
            for score in candidate.scores:
                for example in self.training_data + self.validation_data:
                    if str(example_id(example)) == str(score.id):
                        self.evaluation_cache.put(candidate, example, score)
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
        self._selector = self.config.selection(config=self.config)
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
            restored.values(), key=lambda candidate: candidate.total_score(), default=None
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
                        or selected.total_score() >= candidate.total_score()
                    )
                ):
                    candidate = selected
            except (RuntimeError, ValueError):
                # A selector may have no accumulated scores when compilation
                # ends before the first evaluation completes.
                pass
        return candidate

    def evaluate(self):
        """Evaluate current candidates."""
        if self.config.verbose:
            self.notify(
                "log", "info",
                f"Evaluating candidates in generation {self.current_generation}",
            )

        if self.current_newborns is None or self.current_newborns.is_empty():
            # No candidates to evaluate, move to selection
            self.algorithm_state = "select"
            return

        # Evaluate the newborns (including initial candidate)
        self.current_survivors = self.evaluator.evaluate(self.current_newborns, self.budget)
        self._write_proposal_trace(
            getattr(self.evaluator, "last_proposal_records", [])
        )

        survivors = set(self.current_survivors.candidates)
        for candidate in self.current_newborns:
            self.notify("candidate_evaluated", candidate, candidate in survivors)

        generation_best_score = None

        # Update best candidate tracking
        if self.current_survivors and not self.current_survivors.is_empty():
            best_in_generation = max(self.current_survivors.candidates,
                                   key=lambda c: c.total_score())
            generation_best_score = best_in_generation.average_score()

            if (self.best_candidate is None or
                best_in_generation.total_score() > self.best_candidate.total_score()):
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

    def _write_proposal_trace(self, records: list[dict]) -> None:
        """Append compact proposal decisions to the optional JSONL sink."""
        path = self.config.proposal_trace_path
        if not path or not records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({
                    "generation": self.current_generation,
                    **record,
                }, default=str) + "\n")

    def select(self):
        """Select candidates for next generation."""
        if self.config.verbose:
            self.notify(
                "log", "info",
                f"Selecting candidates for generation {self.current_generation + 1}",
            )

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
            self.notify("finish_iteration", self.current_generation, self.current_parents, self.budget)
            self._iteration_started = False
        self.algorithm_state = "generate"
        self._write_checkpoint()

    def generate(self):
        """Generate new candidates."""
        self.current_generation += 1

        if self.config.verbose:
            self.notify(
                "log", "info",
                f"Generating candidates for generation {self.current_generation}",
            )

        if self.current_parents is None or self.current_parents.is_empty():
            # No parents available, terminate by setting state that leads to exit
            self.algorithm_state = "terminate"
            return

        self.notify("start_iteration", self.current_generation, self.current_parents, self.budget)
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
                sampling_strategy=self.config.sampling_strategy,
                batch_sampler=self.batch_sampler,
                rng=self.rng,
            )

        if self.current_newborns.is_empty() and self.budget <= 0:
            self.notify("budget_exhausted", self.budget)

        # Cycle back to evaluation
        self.algorithm_state = "evaluate"
        self._write_checkpoint()


class GEPAStrategy(BaseStrategy[Result]):
    """GEPA strategy that assembles and delegates to ``GEPAWorkflow``."""

    def __init__(self, config: "GEPAConfig") -> None:
        super().__init__(config)
        object.__setattr__(
            self,
            "workflow",
            GEPAWorkflow(config, notify=self._notify),
        )

    @property
    def student(self):
        """Expose the active student for the base lifecycle notification."""
        return self.workflow.student

    @property
    def dataset_manager(self):
        """Expose the active dataset manager for the base lifecycle notification."""
        return self.workflow.dataset_manager

    def _start_compilation(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        devset: list[dspy.Example] | None = None,
        teacher: dspy.Module | None = None,
        **kwargs,
    ) -> None:
        """Delegate compilation setup to the configured workflow."""
        self.workflow.start_compilation(
            student,
            trainset=trainset,
            devset=devset,
            teacher=teacher,
            **kwargs,
        )

    def _next_step(self) -> bool:
        """Delegate one execution step to the workflow."""
        return self.workflow.next_step()

    def _finish_compilation(self) -> Result:
        """Delegate finalization to the workflow."""
        return self.workflow.finish_compilation()
