import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature
from dspy.teleprompt.bootstrap_finetune import FinetuneTeleprompter
from dspy.teleprompt.utils import create_minibatch, eval_candidate_program, get_signature, set_signature

logger = logging.getLogger(__name__)


# Core Data Structures
@dataclass
class FeedbackResult:
    """Enhanced feedback with traces and diagnostics for reflective mutation."""
    traces: List[List]  # DSPy traces: List of (predictor, inputs, outputs) tuples
    diagnostics: List[str]  # Textual diagnostic feedback
    scores: List[float]  # Scalar scores for each example
    
    
@dataclass 
class BudgetTracker:
    """Track rollout budget to match paper's efficiency claims."""
    used: int = 0
    limit: int = 0
    minibatch_rollouts: int = 0
    reflection_rollouts: int = 0
    pareto_rollouts: int = 0
    
    def add_minibatch_cost(self, cost: int):
        self.minibatch_rollouts += cost
        self.used += cost
        
    def add_reflection_cost(self, cost: int = 1):
        self.reflection_rollouts += cost
        self.used += cost
        
    def add_pareto_cost(self, cost: int):
        self.pareto_rollouts += cost
        self.used += cost
        
    def has_budget(self) -> bool:
        return self.used < self.limit
        
    def get_stats(self) -> Dict[str, int]:
        return {
            'total_used': self.used,
            'minibatch': self.minibatch_rollouts,
            'reflection': self.reflection_rollouts, 
            'pareto': self.pareto_rollouts,
            'remaining': self.limit - self.used
        }


class ScoreMatrix:
    """Manages candidate scores across Pareto evaluation set."""
    
    def __init__(self):
        self.scores: Dict[int, Dict[int, float]] = defaultdict(dict)  # candidate_idx -> task_idx -> score
        
    def set_score(self, candidate_idx: int, task_idx: int, score: float):
        self.scores[candidate_idx][task_idx] = score
        
    def get_score(self, candidate_idx: int, task_idx: int) -> Optional[float]:
        return self.scores.get(candidate_idx, {}).get(task_idx)
        
    def get_candidate_scores(self, candidate_idx: int) -> Dict[int, float]:
        return self.scores.get(candidate_idx, {})
        
    def get_all_candidates(self) -> List[int]:
        return list(self.scores.keys())
        
    def compute_average_score(self, candidate_idx: int) -> float:
        candidate_scores = self.get_candidate_scores(candidate_idx)
        if not candidate_scores:
            return 0.0
        return sum(candidate_scores.values()) / len(candidate_scores)


# Component Interfaces
class CandidateSelector(ABC):
    """Interface for candidate selection strategies (Algorithm 2 from paper)."""
    
    @abstractmethod
    def select_candidate(self, candidates: List[Module], scores: ScoreMatrix) -> int:
        """Select candidate index using selection strategy."""
        raise NotImplementedError


class PromptMutator(ABC):
    """Interface for reflective prompt mutation (core GEPA innovation)."""
    
    @abstractmethod
    def mutate_signature(self, current_signature: Signature, feedback: FeedbackResult) -> Signature:
        """Mutate signature based on reflective feedback."""
        raise NotImplementedError


class FeedbackCollector(ABC):
    """Interface for enhanced feedback collection (μf function)."""
    
    @abstractmethod 
    def collect_feedback(self, program: Module, examples: List[Example], metric: Callable) -> FeedbackResult:
        """Collect enhanced feedback with traces and diagnostics."""
        raise NotImplementedError


class ModuleSelector(ABC):
    """Interface for module selection within programs."""
    
    @abstractmethod
    def select_module(self, program: Module) -> int:
        """Select module index to mutate."""
        raise NotImplementedError


# Concrete Implementations (Stubs)
class ParetoCandidateSelector(CandidateSelector):
    """Pareto-based candidate selection (Algorithm 2)."""
    
    def select_candidate(self, candidates: List[Module], scores: ScoreMatrix) -> int:
        # TODO: Implement Algorithm 2 from paper
        raise NotImplementedError("Pareto selection not yet implemented")


class ReflectivePromptMutator(PromptMutator):
    """Reflective prompt mutation using LLM feedback."""
    
    def __init__(self, prompt_model: Optional[Any] = None):
        self.prompt_model = prompt_model
        
    def mutate_signature(self, current_signature: Signature, feedback: FeedbackResult) -> Signature:
        # TODO: Implement meta-prompt based reflection
        raise NotImplementedError("Reflective mutation not yet implemented")


class EnhancedFeedbackCollector(FeedbackCollector):
    """Enhanced feedback with diagnostic traces."""
    
    def collect_feedback(self, program: Module, examples: List[Example], metric: Callable) -> FeedbackResult:
        # TODO: Implement enhanced feedback with traces
        raise NotImplementedError("Enhanced feedback collection not yet implemented")


class RoundRobinModuleSelector(ModuleSelector):
    """Round-robin module selection strategy."""
    
    def __init__(self):
        self.module_counts = defaultdict(int)
        
    def select_module(self, program: Module) -> int:
        # TODO: Implement round-robin selection
        raise NotImplementedError("Round-robin selection not yet implemented")


# Main GEPA Implementation
class GEPA(FinetuneTeleprompter):
    """
    GEPA: Genetic-Pareto optimizer for compound AI systems.
    
    Based on paper: "GEPA: Reflective Prompt Evolution and Pareto-based Selection"
    Implements Algorithm 1 with Pareto-based candidate selection (Algorithm 2).
    """
    
    def __init__(
        self,
        metric: Callable,
        minibatch_size: int = 3,
        pareto_ratio: float = 0.67,
        merge_enabled: bool = False,
        merge_frequency: int = 5,
        max_errors: Optional[int] = None,
        num_threads: Optional[int] = None,
        prompt_model: Optional[Any] = None,
        candidate_selector: Optional[CandidateSelector] = None,
        prompt_mutator: Optional[PromptMutator] = None,
        feedback_collector: Optional[FeedbackCollector] = None,
        module_selector: Optional[ModuleSelector] = None,
    ):
        super().__init__()
        self.metric = metric
        self.minibatch_size = minibatch_size
        self.pareto_ratio = pareto_ratio
        self.merge_enabled = merge_enabled
        self.merge_frequency = merge_frequency
        self.max_errors = max_errors
        self.num_threads = num_threads
        self.prompt_model = prompt_model or dspy.settings.lm
        
        # Dependency injection with defaults
        self.candidate_selector = candidate_selector or ParetoCandidateSelector()
        self.prompt_mutator = prompt_mutator or ReflectivePromptMutator(self.prompt_model)
        self.feedback_collector = feedback_collector or EnhancedFeedbackCollector()
        self.module_selector = module_selector or RoundRobinModuleSelector()
        
    def compile(
        self, 
        student: Module, 
        *, 
        trainset: List[Example], 
        teacher: Optional[Module] = None, 
        valset: Optional[List[Example]] = None, 
        **kwargs
    ) -> Module:
        """
        Main GEPA compilation implementing Algorithm 1 from paper.
        
        Args:
            student: Program to optimize
            trainset: Training examples
            teacher: Optional teacher program (unused in GEPA)
            valset: Optional validation set
            **kwargs: Additional arguments including 'budget'
            
        Returns:
            Optimized program
        """
        # 1. Validate inputs following DSPy conventions
        if getattr(student, "_compiled", False):
            raise ValueError("Student program should not be pre-compiled")
            
        logger.info("Starting GEPA compilation...")
        
        # 2. Create working copy (never modify original)
        program = student.deepcopy()
        
        # 3. Split dataset (GEPA-specific: feedback vs pareto data)
        feedback_data, pareto_data = self._split_dataset(trainset)
        logger.info(f"Split dataset: {len(feedback_data)} feedback, {len(pareto_data)} pareto examples")
        
        # 4. Initialize GEPA state
        candidates = [program]
        scores = ScoreMatrix()
        budget = BudgetTracker(limit=kwargs.get('budget', len(trainset) * 10))
        
        # 5. Initial Pareto evaluation
        logger.info("Performing initial Pareto evaluation...")
        self._evaluate_candidates_on_pareto(candidates, pareto_data, scores, budget)
        
        # 6. Main optimization loop (Algorithm 1)
        iteration = 0
        while budget.has_budget():
            iteration += 1
            logger.info(f"GEPA iteration {iteration}, budget: {budget.get_stats()}")
            
            try:
                # Algorithm 1 steps
                candidate_idx = self._select_candidate_step(candidates, scores)
                module_idx = self._select_module_step(candidates[candidate_idx])
                feedback = self._collect_feedback_step(candidates[candidate_idx], feedback_data, budget)
                new_candidate = self._mutate_candidate_step(
                    candidates[candidate_idx], module_idx, feedback, budget
                )
                
                # Evaluate and potentially promote new candidate
                if self._should_promote_candidate(new_candidate, candidates[candidate_idx], feedback_data, budget):
                    candidates.append(new_candidate)
                    logger.info(f"Promoted new candidate (total: {len(candidates)})")
                    self._evaluate_candidates_on_pareto([new_candidate], pareto_data, scores, budget)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                if self.max_errors and iteration > self.max_errors:
                    break
                continue
        
        # 7. Return best candidate
        best_candidate = self._select_best_candidate(candidates, scores)
        best_candidate._compiled = True
        
        logger.info(f"GEPA completed in {iteration} iterations")
        logger.info(f"Final budget usage: {budget.get_stats()}")
        
        return best_candidate
    
    def _split_dataset(self, trainset: List[Example]) -> Tuple[List[Example], List[Example]]:
        """Split dataset into feedback and Pareto evaluation sets."""
        # TODO: Implement dataset splitting based on pareto_ratio
        raise NotImplementedError("Dataset splitting not yet implemented")
    
    def _evaluate_candidates_on_pareto(
        self, 
        candidates: List[Module], 
        pareto_data: List[Example], 
        scores: ScoreMatrix,
        budget: BudgetTracker
    ):
        """Evaluate candidates on Pareto set with async support."""
        # TODO: Implement async Pareto evaluation
        raise NotImplementedError("Pareto evaluation not yet implemented")
    
    def _select_candidate_step(self, candidates: List[Module], scores: ScoreMatrix) -> int:
        """Step 1: Select candidate using Pareto-based strategy."""
        return self.candidate_selector.select_candidate(candidates, scores)
    
    def _select_module_step(self, candidate: Module) -> int:
        """Step 2: Select module to mutate."""
        return self.module_selector.select_module(candidate)
    
    def _collect_feedback_step(
        self, 
        candidate: Module, 
        feedback_data: List[Example], 
        budget: BudgetTracker
    ) -> FeedbackResult:
        """Step 3: Collect feedback on minibatch."""
        minibatch = create_minibatch(feedback_data, self.minibatch_size)
        feedback = self.feedback_collector.collect_feedback(candidate, minibatch, self.metric)
        budget.add_minibatch_cost(len(minibatch) * len(candidate.predictors()))
        return feedback
    
    def _mutate_candidate_step(
        self, 
        candidate: Module, 
        module_idx: int, 
        feedback: FeedbackResult, 
        budget: BudgetTracker
    ) -> Module:
        """Step 4: Create mutated candidate."""
        # Create new candidate with mutated module
        new_candidate = candidate.deepcopy()
        current_predictor = new_candidate.predictors()[module_idx]
        current_signature = get_signature(current_predictor)
        
        # Reflective mutation
        new_signature = self.prompt_mutator.mutate_signature(current_signature, feedback)
        set_signature(current_predictor, new_signature)
        budget.add_reflection_cost(1)
        
        return new_candidate
    
    def _should_promote_candidate(
        self, 
        new_candidate: Module, 
        parent_candidate: Module, 
        feedback_data: List[Example], 
        budget: BudgetTracker
    ) -> bool:
        """Decide whether to promote candidate to full evaluation."""
        # TODO: Implement promotion decision logic
        raise NotImplementedError("Candidate promotion logic not yet implemented")
    
    def _select_best_candidate(self, candidates: List[Module], scores: ScoreMatrix) -> Module:
        """Select best candidate based on average Pareto scores."""
        # TODO: Implement best candidate selection
        raise NotImplementedError("Best candidate selection not yet implemented")
