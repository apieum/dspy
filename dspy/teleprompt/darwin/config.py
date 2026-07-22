"""Configuration classes for Darwin optimization framework."""

from dataclasses import dataclass
from typing import Type, Optional, Tuple, Any

from .budget import Budget, LMCallsBudget
from .selection import Selector, ParetoFrontier
from .generation import Generator, SystemAwareMerge, ReflectivePromptMutation
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
    enhanced_feedback: Optional[Assessor] = F1Score()  # Optional feedback-generating metric
    acceptance_criterion: Any = StrictImprovementAcceptance
    proposal_selection: Any = AllImprovements
    validation_policy: Any = FullEvaluationPolicy
    candidate_selection_strategy: str = "pareto"
    proposals_per_generation: int = 1
    mutation_config: ReflectiveMutationConfig = None

    # System parameters we actually have
    max_lm_calls: int = 100
    max_evaluation_calls: Optional[int] = None
    max_generation_calls: Optional[int] = None
    max_iterations: int = 100
    patience: int = 3
    validation_split: float = 0.2
    minibatch_size: int = 3  # Size of minibatch for quick validation
    seed: int = 1

    # Dataset and lifecycle extension points
    dataset_manager_factory: Any = DefaultDatasetManagerFactory
    observers: Tuple[Any, ...] = ()
    checkpoint_path: Optional[str] = None

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
        if self.proposals_per_generation <= 0:
            raise ValueError("proposals_per_generation must be positive")
