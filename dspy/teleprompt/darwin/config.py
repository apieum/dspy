"""Configuration classes for Darwin optimization framework."""

from dataclasses import dataclass
from typing import Type, Optional, Tuple, Any

from .budget import Budget, LMCallsBudget
from .selection import Selector, ParetoFrontier
from .generation import Generator, SystemAwareMerge, ReflectivePromptMutation
from .generation import SamplingStrategy, BatchSampler
from .generation.config import ReflectiveMutationConfig
from .evaluation import Evaluator, GEPATwoPhasesEval
from .evaluation.metrics import Assessor, F1Score
from .dataset_manager import DefaultDatasetManagerFactory
from .evaluation.acceptance import StrictImprovementAcceptance
from .evaluation.proposal_selection import AllImprovements
from .evaluation.policy import FullEvaluationPolicy


@dataclass
class DarwinConfig:
    """Minimal configuration with only component classes and strategic choices.

    Strategy owns all tactical decisions and instantiates components with appropriate data.
    Config only specifies which classes to use and key strategic choices like metrics.
    """
    # Strategic components for strategy to instantiate (required)
    budget: Type['Budget'] = LMCallsBudget
    selection: Type['Selector'] = ParetoFrontier
    evaluation: Type['Evaluator'] = GEPATwoPhasesEval
    mutation: Type['Generator'] = ReflectivePromptMutation
    fitness_function: Assessor = F1Score()

    # Optional strategic choices
    crossover: Optional[Type['Generator']] = SystemAwareMerge  # Optional crossover generator
    use_merge: bool = False
    max_merge_invocations: int = 5
    enhanced_feedback: Optional[Assessor] = F1Score()  # Optional feedback-generating metric
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
    mutation_config: ReflectiveMutationConfig = None

    # System parameters we actually have
    max_lm_calls: int = 100
    max_evaluation_calls: Optional[int] = None
    max_generation_calls: Optional[int] = None
    # A safety cap protects local/mock runs whose budget accounting is not
    # representative. Production runs should set this high or rely on the
    # LM-call budget as their primary stopping condition.
    max_iterations: Optional[int] = 100
    perfect_score: Optional[float] = 1.0
    skip_perfect_score: bool = True
    patience: int = 3
    validation_split: float = 0.2
    minibatch_size: int = 3  # Size of minibatch for quick validation
    seed: int = 1

    # Dataset and lifecycle extension points
    dataset_manager_factory: Any = DefaultDatasetManagerFactory
    observers: Tuple[Any, ...] = ()
    checkpoint_path: Optional[str] = None
    resume_from: Optional[str] = None
    stoppers: Tuple[Any, ...] = ()

    # Logging
    verbose: bool = False

    def __post_init__(self):
        if self.mutation_config is None:
            self.mutation_config = ReflectiveMutationConfig(minibatch_size=self.minibatch_size)
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
