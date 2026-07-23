"""Configuration classes for Darwin optimization framework."""

from abc import ABC
from typing import Type, Optional, Tuple, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .budget import Budget, LMCallsBudget
from .selection import Selector, ParetoFrontier
from .generation import (
    Generator,
    SystemAwareMerge,
    ReflectivePromptMutation,
    GEPAReflection,
    ReflectionStrategy,
    FeedbackProvider,
    EpochShuffledBatchSampler,
    SingleMutationSampling,
)
from .generation import SamplingStrategy, BatchSampler
from .generation.config import ReflectiveMutationConfig
from .evaluation import Evaluator, GEPATwoPhasesEval
from .evaluation.metrics import Assessor, F1Score
from .dataset_manager import DefaultDatasetManagerFactory
from .evaluation.acceptance import StrictImprovementAcceptance
from .evaluation.proposal_selection import AllImprovements
from .evaluation.policy import FullEvaluationPolicy
from .evaluation.batching import PerCandidateBatchEvaluator, resolve_batch_evaluator


class DarwinConfig(ABC):
    """Abstract configuration contract shared by Darwin components.

    Darwin operators receive a configuration object but must not depend on a
    particular strategy's concrete settings type. Concrete strategies provide
    the fields their components need; ``GEPAConfig`` is the first such
    implementation.

    ``BaseStrategy`` requires the lifecycle fields below for every strategy.
    They are declared here so alternative configuration models can satisfy the
    strategy contract without inheriting GEPA-specific settings.
    """

    budget: Type['Budget']
    observers: Tuple[Any, ...]
    verbose: bool
    checkpoint_path: Optional[str]
    resume_from: Optional[str]
    handle_signals: bool

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

class GEPAConfig(BaseModel, DarwinConfig):
    """Configuration for the GEPA strategy implemented by Darwin.

    Strategy owns all tactical decisions and instantiates components with appropriate data.
    The concrete configuration specifies which classes to use and key
    strategic choices such as metrics.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
    )

    # Strategic components for strategy to instantiate (required)
    budget: Type['Budget'] = LMCallsBudget
    selection: Type['Selector'] = ParetoFrontier
    evaluation: Type['Evaluator'] = GEPATwoPhasesEval
    mutation: Type['Generator'] = ReflectivePromptMutation
    fitness_function: Any = Field(default_factory=F1Score)

    # Optional strategic choices
    crossover: Optional[Type['Generator']] = SystemAwareMerge  # Optional crossover generator
    fallback_mutation: Type['Generator'] = ReflectivePromptMutation
    # The complete GEPA strategy enables opportunistic merging by default.
    # Mutation-only optimizers can opt out explicitly.
    use_merge: bool = True
    max_merge_invocations: int = 5
    # Official GEPA only attempts a merge when the two branches share enough
    # validation support to make the merge comparison meaningful.
    merge_val_overlap_floor: int = 5
    # Number of distinct parent pairs inspected during one merge opportunity.
    merge_pair_attempts: int = 10
    # Optional LM dedicated to reflective prompt proposals. When omitted,
    # reflection uses DSPy's active task LM, preserving the existing default.
    reflection_lm: Optional[Any] = None
    # Reuse the parent rollout collected for reflection during acceptance.
    # This is cheaper and deterministic; disabling it matches implementations
    # that independently rerun the parent during validation.
    reuse_parent_rollouts: bool = True
    enhanced_feedback: Any = Field(default_factory=F1Score)  # Optional feedback-generating metric
    acceptance_criterion: Any = StrictImprovementAcceptance
    proposal_selection: Any = AllImprovements
    validation_policy: Any = FullEvaluationPolicy
    preserve_diversity: bool = False
    archive_capacity: int = 32
    candidate_selection_strategy: str = "pareto"
    frontier_type: str = "instance"
    proposals_per_generation: int = 1
    sampling_strategy: Optional[SamplingStrategy] = None
    batch_sampler: Optional[BatchSampler] = None
    elitist_pruning: bool = False
    selector_observers: Tuple[Any, ...] = ()
    # Optional adapter-level evaluator. It receives a list of
    # (candidate, examples) jobs and returns one metric list per job.
    batch_evaluator: Any = Field(default_factory=PerCandidateBatchEvaluator)
    mutation_config: Optional[ReflectiveMutationConfig] = None

    # System parameters we actually have
    max_lm_calls: int = 100
    max_evaluation_calls: Optional[int] = None
    max_generation_calls: Optional[int] = None
    # A safety cap protects local/mock runs whose budget accounting is not
    # representative. Production runs should set this high or rely on the
    # LM-call budget as their primary stopping condition.
    max_iterations: Optional[int] = 100
    perfect_score: Optional[float] = 1.0
    failure_score: float = 0.0
    skip_perfect_score: bool = True
    # GEPA is budget-driven by default.  Set an integer explicitly to enable
    # an optional no-improvement stopping condition.
    patience: Optional[int] = None
    validation_split: float = 0.2
    minibatch_size: int = 3  # Size of minibatch for quick validation
    seed: int = 1

    # Dataset and lifecycle extension points
    dataset_manager_factory: Any = DefaultDatasetManagerFactory
    observers: Tuple[Any, ...] = ()
    checkpoint_path: Optional[str] = None
    resume_from: Optional[str] = None
    # Optional JSONL sink for compact proposal decisions. Rejected candidate
    # modules are not retained by this diagnostic trace.
    proposal_trace_path: Optional[str] = None
    handle_signals: bool = True
    stoppers: Tuple[Any, ...] = ()

    # Logging
    verbose: bool = False

    @field_validator(
        "acceptance_criterion",
        "proposal_selection",
        "validation_policy",
        mode="before",
    )
    @classmethod
    def materialize_policy(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            raise ValueError(f"{info.field_name} must be configured")
        if isinstance(value, type):
            try:
                return value()
            except TypeError as error:
                raise TypeError(
                    "configured policy classes must have a no-argument constructor; "
                    "provide an instance otherwise"
                ) from error
        return value

    @field_validator("batch_evaluator", mode="before")
    @classmethod
    def materialize_batch_evaluator(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("batch_evaluator must be configured")
        return resolve_batch_evaluator(value)

    @model_validator(mode="after")
    def validate_config(self) -> "GEPAConfig":
        if self.mutation_config is None:
            self.mutation_config = ReflectiveMutationConfig(minibatch_size=self.minibatch_size)
        if self.mutation_config.reflection_strategy is None:
            self.mutation_config.reflection_strategy = GEPAReflection()
        if self.sampling_strategy is None:
            self.sampling_strategy = SingleMutationSampling()
        if self.batch_sampler is None:
            self.batch_sampler = EpochShuffledBatchSampler(
                self.mutation_config.minibatch_size
            )
        if not isinstance(self.failure_score, (int, float)):
            raise TypeError("failure_score must be numeric")
        if self.patience is not None and self.patience < 0:
            raise ValueError("patience must be non-negative or None")
        if self.candidate_selection_strategy not in {
            "pareto", "current_best", "epsilon_greedy", "top_k_pareto"
        }:
            raise ValueError(
                "candidate_selection_strategy must be one of: pareto, "
                "current_best, epsilon_greedy, top_k_pareto"
            )
        if self.frontier_type not in {"instance", "objective", "hybrid", "cartesian"}:
            raise ValueError(
                "frontier_type must be one of: instance, objective, hybrid, cartesian"
            )
        if self.proposals_per_generation <= 0:
            raise ValueError("proposals_per_generation must be positive")
        if self.sampling_strategy is not None and not callable(getattr(self.sampling_strategy, "sample", None)):
            raise TypeError("sampling_strategy must provide a sample(parents, count, rng=...) method")
        if self.batch_sampler is not None and not callable(getattr(self.batch_sampler, "sample", None)):
            raise TypeError("batch_sampler must provide a sample(data, count, rng=...) method")
        if self.archive_capacity <= 0:
            raise ValueError("archive_capacity must be positive")
        if self.max_merge_invocations < 0:
            raise ValueError("max_merge_invocations must be non-negative")
        if self.merge_val_overlap_floor <= 0:
            raise ValueError("merge_val_overlap_floor must be positive")
        if self.merge_pair_attempts <= 0:
            raise ValueError("merge_pair_attempts must be positive")
        return self


GEPAConfig.model_rebuild()
