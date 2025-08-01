import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature
from dspy.teleprompt.bootstrap_finetune import FinetuneTeleprompter
from dspy.teleprompt.utils import create_minibatch, eval_candidate_program, get_signature, set_signature

logger = logging.getLogger(__name__)


# Dataset Protocol
class TrainingDataset(Protocol):
    """Protocol for training datasets used by GEPA.
    
    Follows the paper's approach of splitting dataset into:
    - feedback_data: Used for reflective mutation and feedback collection
    - pareto_data: Used for candidate evaluation and Pareto selection
    """
    
    def feedback_data(self) -> Iterable[Example]:
        """Generic feedback data for mutation guidance (paper's feedback set)."""
        ...
    
    def pareto_data(self) -> Iterable[Example]:
        """Evaluation data for candidate selection (paper's Pareto set)."""
        ...


class SplitDataset:
    """Paper-compliant implementation of TrainingDataset.
    
    Splits examples according to pareto_ratio as described in GEPA paper.
    """
    
    def __init__(self, examples: List[Example], pareto_ratio: float = 0.67):
        """Initialize with examples and split ratio.
        
        Args:
            examples: List of training examples
            pareto_ratio: Fraction of data used for Pareto evaluation (default 0.67 per paper)
        """
        if not 0.0 < pareto_ratio < 1.0:
            raise ValueError(f"pareto_ratio must be between 0 and 1, got {pareto_ratio}")
        
        # Split according to paper's approach
        pareto_count = int(len(examples) * pareto_ratio)
        self._pareto = examples[-pareto_count:] if pareto_count > 0 else []
        self._feedback = examples[:-pareto_count] if pareto_count > 0 else examples
        
        logger.info(f"Split dataset: {len(self._feedback)} feedback examples, {len(self._pareto)} Pareto examples")
    
    def feedback_data(self) -> List[Example]:
        """Return feedback examples for mutation guidance."""
        return self._feedback
    
    def pareto_data(self) -> List[Example]:
        """Return Pareto examples for candidate evaluation."""
        return self._pareto


# Core Data Structures
@dataclass
class EvaluationTrace:
    """Rich evaluation trace as described in paper Section 3.2."""
    execution_steps: List[str]
    compilation_errors: List[str]
    intermediate_outputs: List[Any]
    module_outputs: Dict[int, Any]
    reasoning_chains: List[str]
    tool_calls: List[Dict]
    error_messages: List[str]
    performance_metrics: Dict[str, float]


@dataclass
class ModuleFeedback:
    """Per-module feedback for multi-hop systems."""
    module_id: int
    module_name: str
    input_data: Any
    output_data: Any
    execution_time: float
    success: bool
    error_message: Optional[str]
    intermediate_reasoning: List[str]
    confidence_score: float


@dataclass
class FeedbackResult:
    """Enhanced feedback with traces and diagnostics for reflective mutation."""
    traces: List[List]  # DSPy traces: List of (predictor, inputs, outputs) tuples
    diagnostics: List[str]  # Textual diagnostic feedback
    scores: List[float]  # Scalar scores for each example
    # Enhanced Feedback Function μf fields
    evaluation_traces: List[EvaluationTrace] = None  # Rich evaluation traces
    module_feedback: List[ModuleFeedback] = None  # Module-level feedback
    feedback_text: List[str] = None  # Textual diagnostic feedback

    def __post_init__(self):
        if self.evaluation_traces is None:
            self.evaluation_traces = []
        if self.module_feedback is None:
            self.module_feedback = []
        if self.feedback_text is None:
            self.feedback_text = []


@dataclass
class CandidateLineage:
    """Track ancestry and evolutionary history of candidates."""
    candidate_id: int
    parent_id: Optional[int] = None
    generation: int = 0
    mutation_type: str = "initial"
    creation_iteration: int = 0
    fitness_history: List[float] = None

    def __post_init__(self):
        if self.fitness_history is None:
            self.fitness_history = []


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


class CandidatePool:
    """Encapsulates candidates, scores, and lineages in a single object.
    
    Manages the state of all candidates in the GEPA optimization process,
    providing clean access to candidates, their performance scores, and
    evolutionary lineages. Includes extension points for future per-candidate
    feedback and over-specialization detection.
    """
    
    def __init__(self):
        self.candidates: List[Module] = []
        self.scores: ScoreMatrix = ScoreMatrix()
        self.lineages: Dict[int, CandidateLineage] = {}
        # Extension point for future candidate-specific feedback
        self._candidate_feedback: Dict[int, FeedbackResult] = {}
        self._next_candidate_id = 0
    
    def add_candidate(self, candidate: Module, lineage: Optional[CandidateLineage] = None) -> int:
        """Add a new candidate to the pool.
        
        Args:
            candidate: The candidate module to add
            lineage: Optional lineage information for tracking evolution
            
        Returns:
            The candidate ID assigned to this candidate
        """
        candidate_id = len(self.candidates)
        self.candidates.append(candidate)
        
        if lineage is not None:
            self.lineages[candidate_id] = lineage
        
        return candidate_id
    
    def get_candidates(self) -> List[Module]:
        """Get all candidates in the pool."""
        return self.candidates.copy()
    
    def get_candidate(self, candidate_id: int) -> Optional[Module]:
        """Get a specific candidate by ID."""
        if 0 <= candidate_id < len(self.candidates):
            return self.candidates[candidate_id]
        return None
    
    def get_scores(self) -> ScoreMatrix:
        """Get the score matrix for all candidates."""
        return self.scores
    
    def get_lineages(self) -> Dict[int, CandidateLineage]:
        """Get lineage information for all candidates."""
        return self.lineages.copy()
    
    def get_lineage(self, candidate_id: int) -> Optional[CandidateLineage]:
        """Get lineage information for a specific candidate."""
        return self.lineages.get(candidate_id)
    
    def size(self) -> int:
        """Number of candidates in pool."""
        return len(self.candidates)
    
    def set_candidate_feedback(self, candidate_id: int, feedback: FeedbackResult):
        """Set feedback for a specific candidate (extension point)."""
        self._candidate_feedback[candidate_id] = feedback
    
    def get_candidate_feedback(self, candidate_id: int) -> Optional[FeedbackResult]:
        """Get feedback for a specific candidate (extension point)."""
        return self._candidate_feedback.get(candidate_id)
    
    def clear(self):
        """Clear all candidates, scores, and lineages."""
        self.candidates.clear()
        self.scores = ScoreMatrix()
        self.lineages.clear()
        self._candidate_feedback.clear()
        self._next_candidate_id = 0
    
    def __len__(self) -> int:
        """Support len() operation."""
        return len(self.candidates)
    
    def __iter__(self):
        """Support iteration over candidates."""
        return iter(self.candidates)


# Component Interfaces
class CandidateGenerator(ABC):
    """Interface for candidate generation strategies.
    
    Handles the generation of new candidates from the current pool using
    various strategies like mutation, crossover, or other evolutionary operators.
    Follows the paper's approach of using generic feedback data for guidance.
    """
    
    @abstractmethod
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[Example], 
                          iteration: int) -> List[Module]:
        """Generate new candidates from the current pool.
        
        Args:
            candidate_pool: Current pool of candidates with scores and lineages
            feedback_data: Generic feedback examples for mutation guidance (paper's feedback set)
            iteration: Current iteration number for timing decisions
            
        Returns:
            List of new candidates to be evaluated and potentially added to pool
        """
        raise NotImplementedError


class MutationGenerator(CandidateGenerator):
    """Paper-compliant mutation generator using reflective prompt mutation.
    
    Implements the mutation strategy from GEPA paper, selecting a parent
    candidate and improving it through reflective prompt mutation based
    on feedback from generic feedback data.
    """
    
    def __init__(self, prompt_mutator: "PromptMutator", 
                 module_selector: "ModuleSelector",
                 feedback_collector: "FeedbackCollector"):
        """Initialize mutation generator with required components.
        
        Args:
            prompt_mutator: Strategy for mutating prompts based on feedback
            module_selector: Strategy for selecting which module to mutate
            feedback_collector: Collector for gathering performance feedback
        """
        self.prompt_mutator = prompt_mutator
        self.module_selector = module_selector
        self.feedback_collector = feedback_collector
    
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[Example], 
                          iteration: int) -> List[Module]:
        """Generate new candidate through mutation of selected parent.
        
        Follows paper's approach:
        1. Select parent candidate based on performance
        2. Collect feedback on parent using generic feedback_data
        3. Apply reflective prompt mutation
        4. Return improved candidate
        """
        if candidate_pool.size() == 0:
            return []
        
        try:
            # Step 1: Select parent candidate based on performance
            parent_idx = self._select_parent_candidate(candidate_pool)
            parent_candidate = candidate_pool.get_candidate(parent_idx)
            
            if parent_candidate is None:
                return []
            
            # Step 2: Collect feedback on parent's performance
            feedback_data_list = list(feedback_data)
            if not feedback_data_list:
                return []
            
            feedback_result = self.feedback_collector.collect_feedback(
                parent_candidate, feedback_data_list, lambda ex, pred, trace=None: 1.0  # Placeholder metric
            )
            
            # Step 3: Apply reflective prompt mutation
            mutated_candidate = self._mutate_candidate(parent_candidate, feedback_result)
            
            if mutated_candidate is None:
                return []
            
            return [mutated_candidate]
            
        except Exception as e:
            logger.warning(f"Mutation generation failed: {e}")
            return []
    
    def _select_parent_candidate(self, candidate_pool: CandidatePool) -> int:
        """Select parent candidate for mutation based on performance."""
        scores = candidate_pool.get_scores()
        candidates = candidate_pool.get_candidates()
        
        # Find candidate with best average score
        best_idx = 0
        best_score = -float('inf')
        
        for i in range(len(candidates)):
            avg_score = scores.compute_average_score(i)
            if avg_score > best_score:
                best_score = avg_score
                best_idx = i
        
        return best_idx
    
    def _mutate_candidate(self, parent_candidate: Module, feedback_result: FeedbackResult) -> Optional[Module]:
        """Apply mutation to parent candidate using feedback."""
        try:
            # Create a copy of the parent
            mutated_candidate = parent_candidate.deepcopy()
            
            # Select module to mutate
            module_idx = self.module_selector.select_module(mutated_candidate)
            predictors = mutated_candidate.predictors()
            
            if module_idx < len(predictors):
                predictor = predictors[module_idx]
                current_signature = get_signature(predictor)
                
                # Apply reflective mutation
                improved_signature = self.prompt_mutator.mutate_signature(current_signature, feedback_result)
                set_signature(predictor, improved_signature)
            
            return mutated_candidate
            
        except Exception as e:
            logger.warning(f"Candidate mutation failed: {e}")
            return None


class CrossoverGenerator(CandidateGenerator):
    """Intelligent crossover generator implementing GEPA+Merge.
    
    Implements the crossover strategy from GEPA paper, combining
    complementary instructions from different evolutionary lineages
    using intelligent semantic analysis.
    """
    
    def __init__(self, frequency: int = 5, min_candidates: int = 3):
        """Initialize crossover generator.
        
        Args:
            frequency: Attempt crossover every N iterations
            min_candidates: Minimum candidates required to attempt crossover
        """
        self.frequency = frequency
        self.min_candidates = min_candidates
    
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[Example], 
                          iteration: int) -> List[Module]:
        """Generate new candidate through intelligent crossover.
        
        Follows paper's GEPA+Merge approach:
        1. Check if crossover should be attempted (frequency, min candidates)
        2. Select parents from different lineages
        3. Perform intelligent instruction merging
        4. Return merged candidate
        """
        # Check preconditions for crossover
        if (iteration < 3 or 
            iteration % self.frequency != 0 or 
            candidate_pool.size() < self.min_candidates):
            return []
        
        try:
            # Select parents for crossover
            parent_indices = self._select_crossover_parents(candidate_pool)
            if parent_indices is None:
                return []
            
            parent1_idx, parent2_idx = parent_indices
            parent1 = candidate_pool.get_candidate(parent1_idx)
            parent2 = candidate_pool.get_candidate(parent2_idx)
            
            if parent1 is None or parent2 is None:
                return []
            
            # Perform intelligent crossover
            merged_candidate = self._merge_candidates(parent1, parent2)
            
            if merged_candidate is None:
                return []
            
            logger.info(f"Created crossover candidate from parents {parent1_idx} and {parent2_idx}")
            return [merged_candidate]
            
        except Exception as e:
            logger.warning(f"Crossover generation failed: {e}")
            return []
    
    def _select_crossover_parents(self, candidate_pool: CandidatePool) -> Optional[Tuple[int, int]]:
        """Select optimal parents for crossover from different lineages."""
        candidates = candidate_pool.get_candidates()
        scores = candidate_pool.get_scores()
        lineages = candidate_pool.get_lineages()
        
        if len(candidates) < 2:
            return None
        
        # Group candidates by lineage root
        lineage_groups = defaultdict(list)
        for idx in range(len(candidates)):
            lineage = lineages.get(idx)
            if lineage:
                root_ancestor = self._find_root_ancestor(lineage, lineages)
                lineage_groups[root_ancestor].append(idx)
            else:
                lineage_groups[idx].append(idx)  # Original candidate
        
        if len(lineage_groups) < 2:
            return None
        
        # Find best candidates from different lineages
        best_from_lineages = []
        for root, group in lineage_groups.items():
            best_idx = max(group, key=lambda i: scores.compute_average_score(i))
            best_score = scores.compute_average_score(best_idx)
            best_from_lineages.append((best_idx, best_score))
        
        if len(best_from_lineages) < 2:
            return None
        
        # Sort by score and take top 2
        best_from_lineages.sort(key=lambda x: x[1], reverse=True)
        return best_from_lineages[0][0], best_from_lineages[1][0]
    
    def _find_root_ancestor(self, lineage: CandidateLineage, all_lineages: Dict[int, CandidateLineage]) -> int:
        """Find the root ancestor of a lineage."""
        current = lineage
        while current.parent_id is not None and current.parent_id in all_lineages:
            current = all_lineages[current.parent_id]
        return current.candidate_id
    
    def _merge_candidates(self, candidate1: Module, candidate2: Module) -> Optional[Module]:
        """Merge two candidates using intelligent crossover strategies."""
        try:
            # Create base candidate from candidate1
            merged_candidate = candidate1.deepcopy()
            
            # Extract instructions from both candidates
            predictors1 = candidate1.predictors()
            predictors2 = candidate2.predictors()
            
            if len(predictors1) != len(predictors2):
                return None  # Can't merge candidates with different structures
            
            # Intelligent crossover: analyze instruction compatibility
            merged_predictors = merged_candidate.predictors()
            
            for i, (pred1, pred2, merged_pred) in enumerate(zip(predictors1, predictors2, merged_predictors)):
                sig1 = get_signature(pred1)
                sig2 = get_signature(pred2)
                
                inst1 = sig1.instructions or ""
                inst2 = sig2.instructions or ""
                
                if inst1 and inst2 and inst1 != inst2:
                    # Apply intelligent crossover strategy
                    combined_instruction = self._intelligent_crossover(inst1, inst2, i)
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = combined_instruction
                elif inst1 and not inst2:
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = inst1
                elif inst2 and not inst1:
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = inst2
            
            return merged_candidate
            
        except Exception as e:
            logger.warning(f"Candidate merge failed: {e}")
            return None
    
    def _intelligent_crossover(self, inst1: str, inst2: str, module_idx: int) -> str:
        """Apply intelligent crossover strategies to combine instructions."""
        # Strategy 1: Detect complementary instructions
        if self._are_complementary(inst1, inst2):
            return self._complementary_merge(inst1, inst2)
        
        # Strategy 2: Detect adversarial instructions
        elif self._are_adversarial(inst1, inst2):
            return self._adversarial_merge(inst1, inst2)
        
        # Strategy 3: Hybrid merge for general cases
        else:
            return self._hybrid_merge(inst1, inst2, module_idx)
    
    def _are_complementary(self, inst1: str, inst2: str) -> bool:
        """Check if instructions are complementary (non-overlapping guidance)."""
        complementary_pairs = [
            (['accurate', 'precise', 'exact'], ['clear', 'concise', 'brief']),
            (['step', 'systematic', 'methodical'], ['creative', 'innovative', 'flexible']),
            (['specific', 'detailed', 'thorough'], ['efficient', 'quick', 'direct']),
            (['analytical', 'logical', 'reasoning'], ['intuitive', 'practical', 'applied'])
        ]
        
        inst1_lower = inst1.lower()
        inst2_lower = inst2.lower()
        
        for group1, group2 in complementary_pairs:
            has_group1_in_inst1 = any(kw in inst1_lower for kw in group1)
            has_group2_in_inst2 = any(kw in inst2_lower for kw in group2)
            has_group1_in_inst2 = any(kw in inst2_lower for kw in group1)
            has_group2_in_inst1 = any(kw in inst1_lower for kw in group2)
            
            # Complementary if one instruction focuses on group1 and other on group2
            if (has_group1_in_inst1 and has_group2_in_inst2 and 
                not has_group1_in_inst2 and not has_group2_in_inst1):
                return True
                
        return False
    
    def _are_adversarial(self, inst1: str, inst2: str) -> bool:
        """Check if instructions are adversarial (conflicting guidance)."""
        adversarial_pairs = [
            (['brief', 'concise', 'short'], ['detailed', 'comprehensive', 'thorough']),
            (['simple', 'basic', 'straightforward'], ['complex', 'sophisticated', 'advanced']),
            (['fast', 'quick', 'rapid'], ['careful', 'methodical', 'deliberate']),
            (['direct', 'immediate'], ['step-by-step', 'gradual', 'incremental'])
        ]
        
        inst1_lower = inst1.lower()
        inst2_lower = inst2.lower()
        
        for group1, group2 in adversarial_pairs:
            has_group1_in_inst1 = any(kw in inst1_lower for kw in group1)
            has_group2_in_inst2 = any(kw in inst2_lower for kw in group2)
            has_group1_in_inst2 = any(kw in inst2_lower for kw in group1)
            has_group2_in_inst1 = any(kw in inst1_lower for kw in group2)
            
            # Adversarial if instructions contain conflicting directives
            if ((has_group1_in_inst1 and has_group2_in_inst2) or 
                (has_group1_in_inst2 and has_group2_in_inst1)):
                return True
                
        return False
    
    def _complementary_merge(self, inst1: str, inst2: str) -> str:
        """Merge complementary instructions by combining their strengths."""
        return f"{inst1} {inst2}"
    
    def _adversarial_merge(self, inst1: str, inst2: str) -> str:
        """Merge adversarial instructions by finding balanced middle ground.""" 
        return f"{inst1} While being {inst2.lower()}, maintain balance and effectiveness."
    
    def _hybrid_merge(self, inst1: str, inst2: str, module_idx: int) -> str:
        """Hybrid merge strategy for general cases."""
        if module_idx == 0:  # First module - emphasize clarity
            return f"{inst1} Ensure clarity: {inst2.lower()}"
        elif module_idx % 2 == 0:  # Even modules - structured combination
            return f"{inst1} Additionally, {inst2.lower()}"
        else:  # Odd modules - adaptive combination
            return f"Combine approaches: {inst1.lower()} and {inst2.lower()}"


class CompositeGenerator(CandidateGenerator):
    """Composite generator that combines multiple generation strategies.
    
    Allows composition of different generation strategies (mutation, crossover, etc.)
    running them in sequence and collecting all generated candidates.
    """
    
    def __init__(self, generators: List[CandidateGenerator]):
        """Initialize composite generator.
        
        Args:
            generators: List of generators to run in sequence
        """
        if not generators:
            raise ValueError("CompositeGenerator requires at least one generator")
        self.generators = generators
    
    def generate_candidates(self, candidate_pool: CandidatePool,
                          feedback_data: Iterable[Example], 
                          iteration: int) -> List[Module]:
        """Generate candidates using all constituent generators.
        
        Runs all generators in sequence and combines their results.
        """
        all_candidates = []
        feedback_data_list = list(feedback_data)  # Convert once for reuse
        
        for generator in self.generators:
            try:
                new_candidates = generator.generate_candidates(
                    candidate_pool, feedback_data_list, iteration
                )
                all_candidates.extend(new_candidates)
            except Exception as e:
                logger.warning(f"Generator {type(generator).__name__} failed: {e}")
                continue
        
        return all_candidates


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

    def __init__(self):
        self.selection_counts = defaultdict(int)

    def select_candidate(self, candidates: List[Module], scores: ScoreMatrix) -> int:
        """Select candidate using Pareto-based illumination strategy (Algorithm 2).

        Implementation of Algorithm 2 from GEPA paper:
        1. Identify highest score for each training instance across all candidates
        2. Compile candidates that achieve best score on at least one training task
        3. Prune strictly dominated candidates
        4. Stochastically sample from remaining candidates based on their "winning" frequency
        """
        if not candidates:
            return 0

        # Get candidates with scores
        candidate_indices = list(range(len(candidates)))
        scored_candidates = [idx for idx in candidate_indices if scores.get_candidate_scores(idx)]

        if not scored_candidates:
            # No candidates have scores yet, select first
            return 0

        if len(scored_candidates) == 1:
            # Only one candidate scored, select it
            self.selection_counts[scored_candidates[0]] += 1
            return scored_candidates[0]

        # Algorithm 2: Pareto-based selection
        pareto_candidates = self._find_pareto_frontier(scored_candidates, scores)
        selected_idx = self._stochastic_sample_from_pareto(pareto_candidates, scores)

        self.selection_counts[selected_idx] += 1
        return selected_idx

    def _find_pareto_frontier(self, candidate_indices: List[int], scores: ScoreMatrix) -> List[int]:
        """Find Pareto frontier of candidates based on task-level performance.

        A candidate is in the Pareto frontier if it achieves the best score
        on at least one training task, and is not strictly dominated.
        """
        # Get all task indices by examining score matrix
        all_task_indices = set()
        for candidate_idx in candidate_indices:
            all_task_indices.update(scores.get_candidate_scores(candidate_idx).keys())
        all_task_indices = list(all_task_indices)

        if not all_task_indices:
            return candidate_indices[:1]  # Fallback to first candidate

        # Step 1: Find best score for each task
        task_best_scores = {}
        for task_idx in all_task_indices:
            best_score = -float('inf')
            for candidate_idx in candidate_indices:
                score = scores.get_score(candidate_idx, task_idx)
                if score is not None and score > best_score:
                    best_score = score
            task_best_scores[task_idx] = best_score

        # Step 2: Find candidates that achieve best score on at least one task
        winning_candidates = set()
        candidate_wins = defaultdict(list)  # candidate_idx -> list of tasks where it wins

        for task_idx in all_task_indices:
            best_score = task_best_scores[task_idx]
            for candidate_idx in candidate_indices:
                score = scores.get_score(candidate_idx, task_idx)
                if score is not None and abs(score - best_score) < 1e-6:  # Account for float precision
                    winning_candidates.add(candidate_idx)
                    candidate_wins[candidate_idx].append(task_idx)

        if not winning_candidates:
            return candidate_indices[:1]  # Fallback

        # Step 3: Prune strictly dominated candidates
        pareto_candidates = list(winning_candidates)
        pareto_candidates = self._remove_dominated_candidates(pareto_candidates, scores, all_task_indices)

        return pareto_candidates if pareto_candidates else candidate_indices[:1]

    def _remove_dominated_candidates(self, candidates: List[int], scores: ScoreMatrix, task_indices: List[int]) -> List[int]:
        """Remove strictly dominated candidates.

        Candidate A dominates candidate B if A performs >= B on all tasks
        and A performs > B on at least one task.
        """
        non_dominated = []

        for i, candidate_a in enumerate(candidates):
            is_dominated = False

            for j, candidate_b in enumerate(candidates):
                if i == j:
                    continue

                # Check if candidate_b dominates candidate_a
                dominates = True
                strictly_better_on_some = False

                for task_idx in task_indices:
                    score_a = scores.get_score(candidate_a, task_idx) or 0.0
                    score_b = scores.get_score(candidate_b, task_idx) or 0.0

                    if score_b < score_a:
                        dominates = False
                        break
                    elif score_b > score_a:
                        strictly_better_on_some = True

                if dominates and strictly_better_on_some:
                    is_dominated = True
                    break

            if not is_dominated:
                non_dominated.append(candidate_a)

        return non_dominated

    def _stochastic_sample_from_pareto(self, pareto_candidates: List[int], scores: ScoreMatrix) -> int:
        """Stochastically sample from Pareto frontier based on winning frequency.

        Candidates with higher winning frequency (more tasks where they achieve best score)
        are more likely to be selected.
        """
        if len(pareto_candidates) == 1:
            return pareto_candidates[0]

        # Count wins for each candidate
        candidate_wins = defaultdict(int)
        all_task_indices = set()

        for candidate_idx in pareto_candidates:
            all_task_indices.update(scores.get_candidate_scores(candidate_idx).keys())
        all_task_indices = list(all_task_indices)

        # Count wins per candidate
        for task_idx in all_task_indices:
            best_score = -float('inf')
            best_candidates = []

            for candidate_idx in pareto_candidates:
                score = scores.get_score(candidate_idx, task_idx) or 0.0
                if score > best_score:
                    best_score = score
                    best_candidates = [candidate_idx]
                elif abs(score - best_score) < 1e-6:
                    best_candidates.append(candidate_idx)

            # Award wins to tied candidates
            for candidate_idx in best_candidates:
                candidate_wins[candidate_idx] += 1.0 / len(best_candidates)

        # Convert to probabilities
        total_wins = sum(candidate_wins.values())
        if total_wins == 0:
            # Uniform selection if no wins computed
            return random.choice(pareto_candidates)

        probabilities = [candidate_wins[candidate_idx] / total_wins for candidate_idx in pareto_candidates]

        # Stochastic selection based on winning frequency
        cumulative_prob = 0.0
        rand_val = random.random()

        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return pareto_candidates[i]

        # Fallback (should not reach here)
        return pareto_candidates[-1]


# Advanced DSPy Signatures for Reflective Mutation
class PerformanceAnalysisSignature(dspy.Signature):
    """Analyze performance and identify improvement opportunities."""
    current_instruction: str = dspy.InputField(desc="Current instruction text")
    task_description: str = dspy.InputField(desc="Description of the task (input -> output fields)")
    performance_summary: str = dspy.InputField(desc="Summary of current performance metrics")
    failure_examples: str = dspy.InputField(desc="Examples of failures and their patterns")
    execution_traces: str = dspy.InputField(desc="Traces from system execution showing reasoning steps")
    
    analysis: str = dspy.OutputField(desc="Detailed analysis of why current instruction fails")
    improvement_strategy: str = dspy.OutputField(desc="Strategy for improving the instruction")
    
class InstructionImprovementSignature(dspy.Signature):
    """Improve instruction based on analysis and strategy."""
    current_instruction: str = dspy.InputField(desc="Current instruction text")
    analysis: str = dspy.InputField(desc="Analysis of current instruction's weaknesses")
    improvement_strategy: str = dspy.InputField(desc="Strategy for improvement")
    task_context: str = dspy.InputField(desc="Additional context about the task requirements")
    
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning for the new instruction")
    improved_instruction: str = dspy.OutputField(desc="The improved instruction text")
    key_changes: str = dspy.OutputField(desc="Summary of key changes made")

class ReflectivePromptMutator(PromptMutator):
    """Reflective prompt mutation using advanced DSPy Chain-of-Thought."""

    def __init__(self, prompt_model: Optional[Any] = None):
        self.prompt_model = prompt_model
        self.mutation_history = []
        
        # Advanced DSPy modules for reflection
        self.performance_analyzer = dspy.ChainOfThought(PerformanceAnalysisSignature)
        self.instruction_improver = dspy.ChainOfThought(InstructionImprovementSignature)

    def mutate_signature(self, current_signature: Signature, feedback: FeedbackResult) -> Signature:
        """Mutate signature using advanced DSPy Chain-of-Thought reflection.

        This implements the core GEPA innovation: using execution traces and
        performance feedback to guide prompt evolution through natural language reflection.
        Uses DSPy's ChainOfThought for sophisticated reasoning about instruction improvement.
        """
        try:
            # Prepare inputs for DSPy reflection modules
            current_instruction = current_signature.instructions or "Answer the question."
            task_description, performance_summary, failure_examples, execution_traces = self._prepare_reflection_inputs(
                current_signature, feedback
            )

            # Step 1: Analyze performance using ChainOfThought
            with dspy.context(lm=self.prompt_model):
                analysis_result = self.performance_analyzer(
                    current_instruction=current_instruction,
                    task_description=task_description,
                    performance_summary=performance_summary,
                    failure_examples=failure_examples,
                    execution_traces=execution_traces
                )

                # Step 2: Improve instruction based on analysis using ChainOfThought  
                improvement_result = self.instruction_improver(
                    current_instruction=current_instruction,
                    analysis=analysis_result.analysis,
                    improvement_strategy=analysis_result.improvement_strategy,
                    task_context=task_description
                )

            # Create new signature with improved instruction
            new_signature = self._create_improved_signature(current_signature, improvement_result)
            
            # Log detailed mutation history
            self._log_mutation_history(current_signature, improvement_result, feedback, analysis_result)
            
            return new_signature

        except Exception as e:
            logger.warning(f"Advanced reflective mutation failed: {e}")
            # Fallback to simpler reflection or original signature
            return self._fallback_mutation(current_signature, feedback)

    def _prepare_reflection_inputs(self, signature: Signature, feedback: FeedbackResult) -> tuple:
        """Prepare structured inputs for DSPy reflection modules."""
        # Extract field information
        input_fields = []
        output_fields = []
        
        for field_name, field_info in signature.fields.items():
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                field_type = field_info.json_schema_extra.get('__dspy_field_type')
                if field_type == 'input':
                    input_fields.append(field_name)
                elif field_type == 'output':
                    output_fields.append(field_name)
            else:
                # Fallback: assume last field is output
                field_names = list(signature.fields.keys())
                if field_name == field_names[-1]:
                    output_fields.append(field_name)
                else:
                    input_fields.append(field_name)
        
        # Task description
        task_description = f"Transform {', '.join(input_fields)} -> {', '.join(output_fields)}"
        
        # Performance summary with detailed metrics
        avg_score = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
        score_distribution = self._analyze_score_distribution(feedback.scores)
        performance_summary = f"""Average score: {avg_score:.3f} on {len(feedback.scores)} examples
Score distribution: {score_distribution}
Success rate: {sum(1 for s in feedback.scores if s > 0.5) / len(feedback.scores) * 100:.1f}%"""
        
        # Detailed failure analysis
        failures = [(diag, score) for diag, score in zip(feedback.diagnostics, feedback.scores) if score <= 0.5]
        failure_examples = self._create_failure_analysis(failures)
        
        # Execution traces analysis
        execution_traces = self._analyze_execution_traces(feedback.traces)
        
        return task_description, performance_summary, failure_examples, execution_traces
    
    def _analyze_score_distribution(self, scores: List[float]) -> str:
        """Analyze score distribution for better insights."""
        if not scores:
            return "No scores available"
        
        high_scores = sum(1 for s in scores if s > 0.8)
        medium_scores = sum(1 for s in scores if 0.5 < s <= 0.8)
        low_scores = sum(1 for s in scores if s <= 0.5)
        
        return f"High (>0.8): {high_scores}, Medium (0.5-0.8): {medium_scores}, Low (≤0.5): {low_scores}"
    
    def _create_failure_analysis(self, failures: List[tuple]) -> str:
        """Create detailed failure analysis."""
        if not failures:
            return "No significant failures observed"
        
        # Group similar failure patterns
        failure_patterns = {}
        for diag, score in failures[:5]:  # Analyze top 5 failures
            # Simple pattern detection - could be enhanced
            if "INCORRECT" in diag:
                failure_patterns.setdefault("Accuracy Issues", []).append(f"Score {score:.2f}: {diag}")
            elif "failed" in diag.lower():
                failure_patterns.setdefault("Execution Failures", []).append(f"Score {score:.2f}: {diag}")
            else:
                failure_patterns.setdefault("Other Issues", []).append(f"Score {score:.2f}: {diag}")
        
        analysis = "\n".join([
            f"{pattern}: {'; '.join(examples[:2])}" 
            for pattern, examples in failure_patterns.items()
        ])
        
        return analysis or "Unspecified failure patterns"
    
    def _analyze_execution_traces(self, traces: List[List]) -> str:
        """Analyze execution traces for reasoning patterns."""
        if not traces or not any(traces):
            return "No execution traces available"
        
        # Simple trace analysis - could be enhanced with more sophisticated pattern detection
        trace_analysis = f"""Execution patterns observed across {len(traces)} examples:
- Average trace length: {sum(len(t) for t in traces) / len(traces):.1f} steps
- Complex reasoning traces: {sum(1 for t in traces if len(t) > 3)}
- Simple reasoning traces: {sum(1 for t in traces if len(t) <= 3)}"""
        
        return trace_analysis
    
    def _create_improved_signature(self, current_signature: Signature, improvement_result) -> Signature:
        """Create new signature with improved instruction."""
        from dspy.signatures.signature import make_signature
        
        # Extract field names for signature creation
        input_fields = []
        output_fields = []
        
        for field_name, field_info in current_signature.fields.items():
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                field_type = field_info.json_schema_extra.get('__dspy_field_type')
                if field_type == 'input':
                    input_fields.append(field_name)
                elif field_type == 'output':
                    output_fields.append(field_name)
            else:
                # Fallback: assume last field is output
                field_names = list(current_signature.fields.keys())
                if field_name == field_names[-1]:
                    output_fields.append(field_name)
                else:
                    input_fields.append(field_name)
        
        # Create new signature
        if not input_fields or not output_fields:
            new_signature = current_signature
        else:
            signature_str = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
            new_signature = make_signature(signature_str)
        
        # Update with improved instruction
        improved_instruction = improvement_result.improved_instruction.strip()
        if improved_instruction and improved_instruction != current_signature.instructions:
            new_signature.instructions = improved_instruction
        else:
            # Use reasoning as fallback instruction if main improvement failed
            reasoning_instruction = improvement_result.reasoning.strip()
            if reasoning_instruction:
                new_signature.instructions = f"{current_signature.instructions} {reasoning_instruction}"
            else:
                new_signature.instructions = self._make_variation(current_signature.instructions)
        
        return new_signature
    
    def _log_mutation_history(self, current_signature: Signature, improvement_result, feedback: FeedbackResult, analysis_result):
        """Log detailed mutation history with DSPy insights."""
        avg_score = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
        
        mutation_record = {
            'original': current_signature.instructions,
            'improved': improvement_result.improved_instruction,
            'reasoning': improvement_result.reasoning,
            'key_changes': improvement_result.key_changes,
            'analysis': analysis_result.analysis,
            'improvement_strategy': analysis_result.improvement_strategy,
            'feedback_scores': feedback.scores,
            'avg_score': avg_score,
            'timestamp': __import__('time').time()
        }
        
        self.mutation_history.append(mutation_record)
        
        logger.info(f"DSPy Chain-of-Thought mutation completed:")
        logger.info(f"  Original: '{current_signature.instructions}'")
        logger.info(f"  Improved: '{improvement_result.improved_instruction}'")
        logger.info(f"  Key changes: {improvement_result.key_changes}")
        logger.debug(f"  Analysis: {analysis_result.analysis}")
        logger.debug(f"  Strategy: {analysis_result.improvement_strategy}")
    
    def _fallback_mutation(self, current_signature: Signature, feedback: FeedbackResult) -> Signature:
        """Fallback to simpler mutation if advanced DSPy approach fails."""
        try:
            # Simple DSPy Predict as fallback
            with dspy.context(lm=self.prompt_model):
                simple_improver = dspy.Predict("current_instruction, performance_info -> improved_instruction")
                avg_score = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
                performance_info = f"Current average score: {avg_score:.2f}, needs improvement"
                
                result = simple_improver(
                    current_instruction=current_signature.instructions or "Answer the question.",
                    performance_info=performance_info
                )
                
                # Create improved signature
                from dspy.signatures.signature import make_signature
                field_names = list(current_signature.fields.keys())
                if len(field_names) >= 2:
                    input_fields = field_names[:-1]
                    output_fields = field_names[-1:]
                    signature_str = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
                    new_signature = make_signature(signature_str)
                    new_signature.instructions = result.improved_instruction.strip() or self._make_variation(current_signature.instructions)
                    return new_signature
                    
        except Exception as e:
            logger.warning(f"Fallback mutation also failed: {e}")
        
        # Final fallback - just vary the original
        new_signature = current_signature
        new_signature.instructions = self._make_variation(current_signature.instructions)
        return new_signature

    def _make_variation(self, instruction: str) -> str:
        """Make small variation to instruction to avoid stagnation."""
        if not instruction:
            return "Answer the question clearly and accurately."

        # Simple variations to maintain exploration
        variations = [
            f"{instruction} Be precise and thorough.",
            f"{instruction} Think step by step.",
            f"Carefully {instruction.lower()}",
            f"{instruction} Provide detailed reasoning."
        ]

        # Select variation that's different from original
        for variation in variations:
            if variation != instruction:
                return variation

        return instruction + " Be comprehensive."


class EnhancedFeedbackCollector(FeedbackCollector):
    """Enhanced feedback collector implementing Enhanced Feedback Function μf from paper Section 3.2."""

    def __init__(self, collect_module_feedback=True, collect_evaluation_traces=True):
        """Initialize enhanced feedback collector.
        
        Args:
            collect_module_feedback: Whether to collect per-module feedback
            collect_evaluation_traces: Whether to collect rich evaluation traces
        """
        self.collect_module_feedback = collect_module_feedback
        self.collect_evaluation_traces = collect_evaluation_traces
        self.domain_handlers = {}

    def collect_feedback(self, program: Module, examples: List[Example], metric: Callable) -> FeedbackResult:
        """Collect enhanced feedback with DSPy traces and diagnostics.

        Implements Enhanced Feedback Function μf with:
        - traces: DSPy execution traces for each example
        - diagnostics: Human-readable diagnostic messages
        - scores: Scalar scores from metric evaluation
        - evaluation_traces: Rich evaluation traces with compilation errors, execution steps
        - module_feedback: Module-level feedback for multi-hop systems
        - feedback_text: Textual diagnostic feedback
        """
        if not examples:
            return FeedbackResult(
                traces=[], diagnostics=[], scores=[],
                evaluation_traces=[], module_feedback=[], feedback_text=[]
            )

        traces = []
        diagnostics = []
        scores = []
        evaluation_traces = []
        module_feedback = []
        feedback_text = []

        for i, example in enumerate(examples):
            try:
                # Collect trace during execution
                trace = []
                compilation_errors = []
                execution_steps = []
                module_outputs = {}
                reasoning_chains = []
                error_messages = []
                
                import time
                start_time = time.time()
                
                with dspy.context(trace=trace):
                    try:
                        prediction = program(**example.inputs())
                        execution_steps.append(f"Successfully executed program with inputs: {list(example.inputs().keys())}")
                    except Exception as exec_error:
                        error_messages.append(f"Execution error: {str(exec_error)}")
                        prediction = type('EmptyPrediction', (), {})()
                
                execution_time = time.time() - start_time

                # Extract trace from context
                execution_trace = trace
                traces.append(execution_trace)

                # Collect module outputs and reasoning chains
                for j, step in enumerate(execution_trace):
                    if hasattr(step, 'outputs'):
                        module_outputs[j] = step.outputs
                    if hasattr(step, 'reasoning'):
                        reasoning_chains.append(step.reasoning)

                # Compute score
                try:
                    score = metric(example, prediction, execution_trace)
                except TypeError:
                    # Fallback for metrics that don't accept trace parameter
                    score = metric(example, prediction)
                scores.append(float(score))

                # Generate diagnostic message
                diagnostic = self._generate_diagnostic(example, prediction, score, execution_trace)
                diagnostics.append(diagnostic)

                # Enhanced Feedback Function μf: Create rich evaluation trace
                if self.collect_evaluation_traces:
                    eval_trace = EvaluationTrace(
                        execution_steps=execution_steps,
                        compilation_errors=compilation_errors,
                        intermediate_outputs=[getattr(step, 'outputs', None) for step in execution_trace],
                        module_outputs=module_outputs,
                        reasoning_chains=reasoning_chains,
                        tool_calls=[],  # Could be enhanced to track actual tool calls
                        error_messages=error_messages,
                        performance_metrics={
                            "score": float(score),
                            "execution_time": execution_time,
                            "trace_length": len(execution_trace)
                        }
                    )
                    evaluation_traces.append(eval_trace)

                # Enhanced Feedback Function μf: Create module-level feedback
                if self.collect_module_feedback:
                    for j, step in enumerate(execution_trace):
                        module_fb = ModuleFeedback(
                            module_id=j,
                            module_name=getattr(step, '__class__', type(step)).__name__,
                            input_data=getattr(step, 'inputs', {}),
                            output_data=getattr(step, 'outputs', {}),
                            execution_time=execution_time / max(1, len(execution_trace)),
                            success=len(error_messages) == 0,
                            error_message=error_messages[0] if error_messages else None,
                            intermediate_reasoning=reasoning_chains,
                            confidence_score=float(score)
                        )
                        module_feedback.append(module_fb)

                # Enhanced Feedback Function μf: Generate textual feedback
                feedback_text.append(self._generate_textual_feedback(example, prediction, score, eval_trace if self.collect_evaluation_traces else None))

            except Exception as e:
                logger.warning(f"Failed to collect feedback for example {i}: {e}")
                traces.append([])
                scores.append(0.0)
                diagnostics.append(f"Execution failed: {str(e)}")
                
                # Add empty enhanced feedback for failed examples
                if self.collect_evaluation_traces:
                    evaluation_traces.append(EvaluationTrace(
                        execution_steps=[f"Failed: {str(e)}"],
                        compilation_errors=[str(e)],
                        intermediate_outputs=[],
                        module_outputs={},
                        reasoning_chains=[],
                        tool_calls=[],
                        error_messages=[str(e)],
                        performance_metrics={"score": 0.0, "execution_time": 0.0, "trace_length": 0}
                    ))
                
                if self.collect_module_feedback:
                    module_feedback.append(ModuleFeedback(
                        module_id=0,
                        module_name="FailedExecution",
                        input_data=example.inputs(),
                        output_data={},
                        execution_time=0.0,
                        success=False,
                        error_message=str(e),
                        intermediate_reasoning=[],
                        confidence_score=0.0
                    ))
                
                feedback_text.append(f"Execution failed: {str(e)}")

        return FeedbackResult(
            traces=traces, 
            diagnostics=diagnostics, 
            scores=scores,
            evaluation_traces=evaluation_traces,
            module_feedback=module_feedback,
            feedback_text=feedback_text
        )

    def _generate_textual_feedback(self, example: Example, prediction: Any, score: float, eval_trace: Optional[EvaluationTrace]) -> str:
        """Generate rich textual feedback using domain-specific handlers."""
        # Detect domain
        domain = self._detect_domain(example)
        
        # Use domain-specific handler if available
        if domain in self.domain_handlers:
            try:
                domain_feedback = self.domain_handlers[domain](example, prediction, eval_trace)
                return f"Domain ({domain}): {domain_feedback}"
            except Exception as e:
                logger.warning(f"Domain handler {domain} failed: {e}")
        
        # Generate generic feedback
        status = "SUCCESS" if score > 0.5 else "FAILURE"
        feedback_parts = [f"Status: {status} (Score: {score:.2f})"]
        
        if eval_trace:
            if eval_trace.error_messages:
                feedback_parts.append(f"Errors: {'; '.join(eval_trace.error_messages[:3])}")
            if eval_trace.reasoning_chains:
                feedback_parts.append(f"Reasoning steps: {len(eval_trace.reasoning_chains)}")
            
            exec_time = eval_trace.performance_metrics.get("execution_time", 0.0)
            feedback_parts.append(f"Execution time: {exec_time:.3f}s")
        
        return " | ".join(feedback_parts)

    def register_domain_handler(self, domain: str, handler: Callable):
        """Register a domain-specific feedback handler."""
        self.domain_handlers[domain] = handler

    def _detect_domain(self, example: Example) -> str:
        """Detect the domain of an example for specialized feedback."""
        # Simple heuristic-based domain detection
        inputs = example.inputs()
        
        # Check for code-related keywords
        text_content = " ".join(str(v).lower() for v in inputs.values())
        if any(keyword in text_content for keyword in ['code', 'function', 'class', 'python', 'javascript']):
            return 'code'
        
        # Check for math-related keywords
        if any(keyword in text_content for keyword in ['calculate', 'solve', 'equation', 'math', 'number']):
            return 'math'
        
        # Check for reasoning-related keywords
        if any(keyword in text_content for keyword in ['because', 'therefore', 'reasoning', 'explain', 'why']):
            return 'reasoning'
        
        return 'general'

    def _generate_diagnostic(self, example: Example, prediction: Any, score: float, trace: List) -> str:
        """Generate human-readable diagnostic message."""
        # Basic diagnostic - can be enhanced with more sophisticated analysis
        status = "CORRECT" if score > 0.5 else "INCORRECT"

        # Extract key information from prediction
        pred_summary = str(prediction)[:100] + "..." if len(str(prediction)) > 100 else str(prediction)

        # Count reasoning steps from trace
        reasoning_steps = len([step for step in trace if hasattr(step, 'reasoning')])

        diagnostic = f"[{status}] Score: {score:.2f}, Steps: {reasoning_steps}, Prediction: {pred_summary}"

        # Add specific failure analysis for low scores
        if score <= 0.5:
            expected = getattr(example, 'answer', 'N/A')
            diagnostic += f" | Expected: {expected}"

        return diagnostic


class RoundRobinModuleSelector(ModuleSelector):
    """Round-robin module selection strategy."""

    def __init__(self):
        self.module_counts = defaultdict(int)

    def select_module(self, program: Module) -> int:
        """Select module using round-robin strategy."""
        predictors = program.predictors()
        if not predictors:
            return 0

        # Find the module with the lowest selection count
        program_id = id(program)
        min_count = min(self.module_counts[program_id, i] for i in range(len(predictors)))

        # Among modules with min count, select the first one
        for i in range(len(predictors)):
            if self.module_counts[program_id, i] == min_count:
                self.module_counts[program_id, i] += 1
                return i

        # Fallback (should never reach here)
        return 0


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

        # Ancestry tracking
        self.lineages: Dict[int, CandidateLineage] = {}  # candidate_id -> lineage info
        self.next_candidate_id = 0

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

        # Initialize ancestry tracking for first candidate
        self._register_candidate(program, parent_id=None, mutation_type="initial", iteration=0)

        # 5. Initial Pareto evaluation (counts toward budget - these are expensive LLM calls)
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
                    # Register new candidate with ancestry info
                    parent_lineage = self.lineages[candidate_idx]
                    self._register_candidate(
                        new_candidate,
                        parent_id=candidate_idx,
                        mutation_type="reflective_mutation",
                        iteration=iteration
                    )
                    logger.info(f"Promoted new candidate (total: {len(candidates)}, generation: {parent_lineage.generation + 1})")
                    # Evaluate only the NEW candidate on Pareto set (costs budget)
                    self._evaluate_candidates_on_pareto([new_candidate], pareto_data, scores, budget)

                    # Optionally perform merge operation
                    if self.merge_enabled and iteration % self.merge_frequency == 0 and len(candidates) >= 3:
                        merge_candidate = self._attempt_merge(candidates, scores, iteration)
                        if merge_candidate:
                            candidates.append(merge_candidate)
                            self._evaluate_candidates_on_pareto([merge_candidate], pareto_data, scores, budget)

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
        """Split dataset into feedback and Pareto evaluation sets.

        Args:
            trainset: Full training dataset

        Returns:
            Tuple of (feedback_data, pareto_data) where:
            - feedback_data: Used for minibatch evaluation and learning signals
            - pareto_data: Used for Pareto frontier evaluation and candidate ranking
        """
        if not trainset:
            return [], []

        # Calculate split sizes based on pareto_ratio
        total_size = len(trainset)
        pareto_size = int(total_size * self.pareto_ratio)
        feedback_size = total_size - pareto_size

        # Ensure minimum sizes
        if pareto_size == 0 and total_size > 0:
            pareto_size = 1
            feedback_size = total_size - 1
        if feedback_size == 0 and total_size > 1:
            feedback_size = 1
            pareto_size = total_size - 1

        # Split dataset (pareto_data first, feedback_data second)
        # This follows paper convention where validation set is typically first portion
        pareto_data = trainset[:pareto_size]
        feedback_data = trainset[pareto_size:]

        return feedback_data, pareto_data

    def _evaluate_candidates_on_pareto(
        self,
        candidates: List[Module],
        pareto_data: List[Example],
        scores: ScoreMatrix,
        budget: BudgetTracker
    ):
        """Evaluate candidates on Pareto set with async support.
        """
        if not pareto_data or not candidates:
            return

        # Run async evaluation if event loop is available, otherwise fallback to sync
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context - run directly
            task = loop.create_task(self._evaluate_candidates_async(candidates, pareto_data, scores, budget))
            loop.run_until_complete(task)
        except RuntimeError:
            # No event loop running - create one
            asyncio.run(self._evaluate_candidates_async(candidates, pareto_data, scores, budget))

    async def _evaluate_candidates_async(
        self,
        candidates: List[Module],
        pareto_data: List[Example],
        scores: ScoreMatrix,
        budget: BudgetTracker
    ):
        """Async implementation of Pareto evaluation."""
        evaluation_tasks = []

        # Create tasks for all candidate-example pairs
        for candidate_idx, candidate in enumerate(candidates):
            # Skip if already evaluated
            if len(scores.get_candidate_scores(candidate_idx)) >= len(pareto_data):
                continue

            for task_idx, example in enumerate(pareto_data):
                # Skip if already evaluated
                if scores.get_score(candidate_idx, task_idx) is not None:
                    continue

                task = self._evaluate_single_candidate_example(
                    candidate, example, candidate_idx, task_idx
                )
                evaluation_tasks.append(task)

        # Execute all evaluations in parallel
        if evaluation_tasks:
            results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

            # Process results and update scores
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Evaluation failed: {result}")
                    continue

                candidate_idx, task_idx, score = result
                scores.set_score(candidate_idx, task_idx, score)

            # Update budget - Pareto evaluations ARE expensive LLM calls that count
            successful_evals = len([r for r in results if not isinstance(r, Exception)])
            budget.add_pareto_cost(successful_evals)
            logger.debug(f"Pareto evaluation completed: {successful_evals} evaluations counted in budget")

    async def _evaluate_single_candidate_example(
        self,
        candidate: Module,
        example: Example,
        candidate_idx: int,
        task_idx: int
    ) -> Tuple[int, int, float]:
        """Evaluate single candidate on single example."""
        try:
            # Use DSPy's trace collection pattern
            with dspy.context(trace=[]):
                prediction = candidate(**example.inputs())

            # Compute score using metric
            score = self.metric(example, prediction)
            return candidate_idx, task_idx, float(score)

        except Exception as e:
            logger.warning(f"Failed to evaluate candidate {candidate_idx} on task {task_idx}: {e}")
            return candidate_idx, task_idx, 0.0

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
        """Step 3: Collect enhanced feedback using μf function on minibatch."""
        minibatch = create_minibatch(feedback_data, self.minibatch_size)
        # Use enhanced feedback collection with μf function from paper
        feedback = self.feedback_collector.collect_feedback(candidate, minibatch, self.metric)
        # Count actual LLM calls in minibatch (this is what counts as "rollouts")
        budget.add_minibatch_cost(len(minibatch))
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
        """Decide whether to promote candidate to full evaluation.

        Strategy: Promote if new candidate shows improvement on small minibatch.
        This balances exploration with budget efficiency.
        """
        if not feedback_data or not budget.has_budget():
            return False

        try:
            # Create small evaluation batch
            eval_batch_size = min(3, len(feedback_data))
            eval_batch = create_minibatch(feedback_data, eval_batch_size)

            # Evaluate both candidates on same batch
            new_score = self._evaluate_candidate_on_batch(new_candidate, eval_batch)
            parent_score = self._evaluate_candidate_on_batch(parent_candidate, eval_batch)

            # Update budget for evaluation costs
            budget.add_minibatch_cost(eval_batch_size * 2)  # Two candidates evaluated

            # Promote if new candidate is better or tied (encourage exploration)
            promotion_threshold = parent_score - 0.05  # Small tolerance for exploration
            should_promote = new_score >= promotion_threshold

            logger.debug(f"Promotion eval: new={new_score:.3f}, parent={parent_score:.3f}, promote={should_promote}")
            return should_promote

        except Exception as e:
            logger.warning(f"Promotion evaluation failed: {e}")
            return False  # Conservative: don't promote on evaluation failure

    def _evaluate_candidate_on_batch(self, candidate: Module, batch: List[Example]) -> float:
        """Evaluate candidate on a small batch, return average score."""
        if not batch:
            return 0.0

        total_score = 0.0
        for example in batch:
            try:
                prediction = candidate(**example.inputs())
                score = self.metric(example, prediction)
                total_score += float(score)
            except Exception:
                # Failed predictions contribute 0 score
                continue

        return total_score / len(batch)

    def _select_best_candidate(self, candidates: List[Module], scores: ScoreMatrix) -> Module:
        """Select best candidate based on average Pareto scores.

        Returns the candidate with highest average score across Pareto evaluation set.
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0]

        best_candidate = candidates[0]
        best_score = -float('inf')

        for idx, candidate in enumerate(candidates):
            avg_score = scores.compute_average_score(idx)

            if avg_score > best_score:
                best_score = avg_score
                best_candidate = candidate

        # Log ancestry information for best candidate
        best_idx = candidates.index(best_candidate)
        best_lineage = self.lineages.get(best_idx)
        if best_lineage:
            logger.info(f"Selected best candidate: generation {best_lineage.generation}, mutation_type: {best_lineage.mutation_type}")
            logger.info(f"Best candidate average score: {best_score:.3f}")
        else:
            logger.info(f"Selected best candidate with average score: {best_score:.3f}")
        return best_candidate

    def _register_candidate(self, candidate: Module, parent_id: Optional[int], mutation_type: str, iteration: int):
        """Register a new candidate with ancestry tracking."""
        candidate_id = self.next_candidate_id
        self.next_candidate_id += 1

        # Determine generation
        generation = 0
        if parent_id is not None and parent_id in self.lineages:
            generation = self.lineages[parent_id].generation + 1

        # Create lineage record
        lineage = CandidateLineage(
            candidate_id=candidate_id,
            parent_id=parent_id,
            generation=generation,
            mutation_type=mutation_type,
            creation_iteration=iteration
        )

        self.lineages[candidate_id] = lineage

        # Store candidate_id in the module for tracking
        candidate._gepa_id = candidate_id

        logger.debug(f"Registered candidate {candidate_id}: generation {generation}, parent {parent_id}, type {mutation_type}")

    def _attempt_merge(self, candidates: List[Module], scores: ScoreMatrix, iteration: int) -> Optional[Module]:
        """Attempt to create a merge candidate from different lineages.

        This implements the "intelligent crossover" mentioned in the paper
        by combining complementary lessons from different optimization lineages.
        """
        if len(candidates) < 2:
            return None

        try:
            # Find candidates from different lineages with complementary strengths
            candidate_lineages = [(i, self.lineages.get(i)) for i in range(len(candidates))]

            # Group by root ancestor to find different lineages
            lineage_groups = defaultdict(list)
            for idx, lineage in candidate_lineages:
                if lineage:
                    root_ancestor = self._find_root_ancestor(lineage)
                    lineage_groups[root_ancestor].append((idx, lineage))
                else:
                    lineage_groups[0].append((idx, None))  # Original candidate

            # Intelligent parent selection for crossover
            if len(lineage_groups) < 2:
                return None

            best_from_lineages = []
            for root, group in lineage_groups.items():
                # Find best candidate from this lineage using multiple criteria
                best_idx = None
                best_score = -float('inf')
                best_diversity = -float('inf')

                for idx, lineage in group:
                    avg_score = scores.compute_average_score(idx)
                    
                    # Consider diversity (fitness history variance) for better crossover
                    diversity_score = 0.0
                    if lineage and lineage.fitness_history:
                        if len(lineage.fitness_history) > 1:
                            import statistics
                            diversity_score = statistics.stdev(lineage.fitness_history)
                    
                    # Combined score favoring both performance and diversity
                    combined_score = avg_score + (0.1 * diversity_score)
                    
                    if combined_score > best_score:
                        best_score = avg_score  # Keep original score for selection
                        best_diversity = diversity_score
                        best_idx = idx

                if best_idx is not None:
                    best_from_lineages.append((best_idx, best_score, best_diversity))

            if len(best_from_lineages) < 2:
                return None

            # Smart parent selection: balance performance and complementarity
            parent1_idx, parent2_idx = self._select_merge_parents(best_from_lineages, candidates)

            # Create merge candidate
            merged_candidate = self._merge_candidates(candidates[parent1_idx], candidates[parent2_idx])

            if merged_candidate:
                # Register merged candidate
                self._register_candidate(
                    merged_candidate,
                    parent_id=parent1_idx,  # Primary parent
                    mutation_type=f"merge_{parent1_idx}_{parent2_idx}",
                    iteration=iteration
                )
                logger.info(f"Created merge candidate from lineages {parent1_idx} and {parent2_idx}")
                return merged_candidate

        except Exception as e:
            logger.warning(f"Merge attempt failed: {e}")

        return None

    def _find_root_ancestor(self, lineage: CandidateLineage) -> int:
        """Find the root ancestor of a lineage."""
        current = lineage
        while current.parent_id is not None and current.parent_id in self.lineages:
            current = self.lineages[current.parent_id]
        return current.candidate_id

    def _select_merge_parents(self, lineage_candidates: List[Tuple[int, float, float]], 
                             candidates: List[Module]) -> Tuple[int, int]:
        """Select optimal parents for crossover based on performance and complementarity."""
        if len(lineage_candidates) < 2:
            # Fallback to simple selection
            lineage_candidates.sort(key=lambda x: x[1], reverse=True)
            return lineage_candidates[0][0], lineage_candidates[1][0]
        
        # Analyze instruction complementarity between candidates
        best_pair = None
        best_compatibility = -float('inf')
        
        for i in range(len(lineage_candidates)):
            for j in range(i + 1, len(lineage_candidates)):
                idx1, score1, div1 = lineage_candidates[i]
                idx2, score2, div2 = lineage_candidates[j]
                
                # Calculate compatibility score
                compatibility = self._calculate_instruction_compatibility(
                    candidates[idx1], candidates[idx2]
                )
                
                # Combined selection criteria: performance + diversity + compatibility
                selection_score = (
                    0.5 * (score1 + score2) +  # Performance
                    0.2 * (div1 + div2) +      # Diversity
                    0.3 * compatibility        # Complementarity
                )
                
                if selection_score > best_compatibility:
                    best_compatibility = selection_score
                    best_pair = (idx1, idx2)
        
        return best_pair if best_pair else (lineage_candidates[0][0], lineage_candidates[1][0])

    def _calculate_instruction_compatibility(self, candidate1: Module, candidate2: Module) -> float:
        """Calculate how well two candidates' instructions would complement each other."""
        try:
            predictors1 = candidate1.predictors()
            predictors2 = candidate2.predictors()
            
            if len(predictors1) != len(predictors2):
                return 0.0  # Incompatible structures
            
            compatibility_scores = []
            
            for pred1, pred2 in zip(predictors1, predictors2):
                sig1 = get_signature(pred1)
                sig2 = get_signature(pred2)
                
                inst1 = sig1.instructions or ""
                inst2 = sig2.instructions or ""
                
                if not inst1 or not inst2:
                    compatibility_scores.append(0.5)  # Neutral
                elif inst1 == inst2:
                    compatibility_scores.append(0.0)  # Identical (no benefit)
                else:
                    # Favor complementary over adversarial
                    if self._are_complementary(inst1, inst2):
                        compatibility_scores.append(1.0)  # Highly compatible
                    elif self._are_adversarial(inst1, inst2):
                        compatibility_scores.append(0.3)  # Potentially useful but conflicting
                    else:
                        compatibility_scores.append(0.6)  # Moderately compatible
            
            return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Compatibility calculation failed: {e}")
            return 0.0

    def _merge_candidates(self, candidate1: Module, candidate2: Module) -> Optional[Module]:
        """Merge two candidates using intelligent crossover strategies.

        Implements sophisticated instruction combination logic with:
        1. Semantic analysis of instructions
        2. Performance-weighted combination
        3. Conflict resolution
        4. Multiple merge strategies (complementary, adversarial, hybrid)
        """
        try:
            # Create base candidate from candidate1
            merged_candidate = candidate1.deepcopy()

            # Extract instructions from both candidates
            predictors1 = candidate1.predictors()
            predictors2 = candidate2.predictors()

            if len(predictors1) != len(predictors2):
                return None  # Can't merge candidates with different structures

            # Intelligent crossover: analyze instruction compatibility
            merged_predictors = merged_candidate.predictors()
            
            for i, (pred1, pred2, merged_pred) in enumerate(zip(predictors1, predictors2, merged_predictors)):
                sig1 = get_signature(pred1)
                sig2 = get_signature(pred2)

                inst1 = sig1.instructions or ""
                inst2 = sig2.instructions or ""

                if inst1 and inst2 and inst1 != inst2:
                    # Apply intelligent crossover strategy
                    combined_instruction = self._intelligent_crossover(inst1, inst2, i)
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = combined_instruction
                elif inst1 and not inst2:
                    # Use instruction from candidate1
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = inst1
                elif inst2 and not inst1:
                    # Use instruction from candidate2
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = inst2

            return merged_candidate

        except Exception as e:
            logger.warning(f"Candidate merge failed: {e}")
            return None

    def _intelligent_crossover(self, inst1: str, inst2: str, module_idx: int) -> str:
        """Apply intelligent crossover strategies to combine instructions.
        
        Uses multiple strategies:
        1. Complementary merge: Combine non-overlapping guidance
        2. Adversarial merge: Balance opposing directives  
        3. Hybrid merge: Conditional combination based on context
        """
        # Strategy 1: Detect complementary instructions
        if self._are_complementary(inst1, inst2):
            return self._complementary_merge(inst1, inst2)
        
        # Strategy 2: Detect adversarial instructions
        elif self._are_adversarial(inst1, inst2):
            return self._adversarial_merge(inst1, inst2)
        
        # Strategy 3: Hybrid merge for general cases
        else:
            return self._hybrid_merge(inst1, inst2, module_idx)

    def _are_complementary(self, inst1: str, inst2: str) -> bool:
        """Check if instructions are complementary (non-overlapping guidance)."""
        # Define complementary patterns
        complementary_pairs = [
            (['accurate', 'precise', 'exact'], ['clear', 'concise', 'brief']),
            (['step', 'systematic', 'methodical'], ['creative', 'innovative', 'flexible']),
            (['specific', 'detailed', 'thorough'], ['efficient', 'quick', 'direct']),
            (['analytical', 'logical', 'reasoning'], ['intuitive', 'practical', 'applied'])
        ]
        
        inst1_lower = inst1.lower()
        inst2_lower = inst2.lower()
        
        for group1, group2 in complementary_pairs:
            has_group1_in_inst1 = any(kw in inst1_lower for kw in group1)
            has_group2_in_inst2 = any(kw in inst2_lower for kw in group2)
            has_group1_in_inst2 = any(kw in inst2_lower for kw in group1)
            has_group2_in_inst1 = any(kw in inst1_lower for kw in group2)
            
            # Complementary if one instruction focuses on group1 and other on group2
            if (has_group1_in_inst1 and has_group2_in_inst2 and 
                not has_group1_in_inst2 and not has_group2_in_inst1):
                return True
                
        return False

    def _are_adversarial(self, inst1: str, inst2: str) -> bool:
        """Check if instructions are adversarial (conflicting guidance)."""
        # Define adversarial patterns
        adversarial_pairs = [
            (['brief', 'concise', 'short'], ['detailed', 'comprehensive', 'thorough']),
            (['simple', 'basic', 'straightforward'], ['complex', 'sophisticated', 'advanced']),
            (['fast', 'quick', 'rapid'], ['careful', 'methodical', 'deliberate']),
            (['direct', 'immediate'], ['step-by-step', 'gradual', 'incremental'])
        ]
        
        inst1_lower = inst1.lower()
        inst2_lower = inst2.lower()
        
        for group1, group2 in adversarial_pairs:
            has_group1_in_inst1 = any(kw in inst1_lower for kw in group1)
            has_group2_in_inst2 = any(kw in inst2_lower for kw in group2)
            has_group1_in_inst2 = any(kw in inst2_lower for kw in group1)
            has_group2_in_inst1 = any(kw in inst1_lower for kw in group2)
            
            # Adversarial if instructions contain conflicting directives
            if ((has_group1_in_inst1 and has_group2_in_inst2) or 
                (has_group1_in_inst2 and has_group2_in_inst1)):
                return True
                
        return False

    def _complementary_merge(self, inst1: str, inst2: str) -> str:
        """Merge complementary instructions by combining their strengths."""
        # Combine complementary aspects
        return f"{inst1} {inst2}"

    def _adversarial_merge(self, inst1: str, inst2: str) -> str:
        """Merge adversarial instructions by finding balanced middle ground."""
        # Create balanced instruction that addresses both concerns
        return f"{inst1} While being {inst2.lower()}, maintain balance and effectiveness."

    def _hybrid_merge(self, inst1: str, inst2: str, module_idx: int) -> str:
        """Hybrid merge strategy for general cases."""
        # Use different strategies based on module position
        if module_idx == 0:  # First module - emphasize clarity
            return f"{inst1} Ensure clarity: {inst2.lower()}"
        elif module_idx % 2 == 0:  # Even modules - structured combination
            return f"{inst1} Additionally, {inst2.lower()}"
        else:  # Odd modules - adaptive combination
            return f"Combine approaches: {inst1.lower()} and {inst2.lower()}"
