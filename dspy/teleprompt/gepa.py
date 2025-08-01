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


class ReflectivePromptMutator(PromptMutator):
    """Reflective prompt mutation using LLM feedback."""
    
    def __init__(self, prompt_model: Optional[Any] = None):
        self.prompt_model = prompt_model
        self.mutation_history = []
        
    def mutate_signature(self, current_signature: Signature, feedback: FeedbackResult) -> Signature:
        """Mutate signature using reflective feedback from execution traces.
        
        This implements the core GEPA innovation: using execution traces and
        performance feedback to guide prompt evolution through natural language reflection.
        """
        try:
            # Generate reflection prompt using feedback
            reflection_prompt = self._create_reflection_prompt(current_signature, feedback)
            
            # Get LLM reflection on how to improve the instruction
            with dspy.context(lm=self.prompt_model):
                reflection_predictor = dspy.Predict("analysis, current_instruction -> improved_instruction")
                reflection_result = reflection_predictor(
                    analysis=reflection_prompt,
                    current_instruction=current_signature.instructions or "Answer the question."
                )
            
            # Create new signature with improved instruction
            # Use make_signature to properly create a new signature
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
            
            # Create signature string
            if not input_fields or not output_fields:
                # Fallback to original signature
                new_signature = current_signature
            else:
                signature_str = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
                new_signature = make_signature(signature_str)
            
            # Update instruction
            improved_instruction = reflection_result.improved_instruction.strip()
            if improved_instruction and improved_instruction != current_signature.instructions:
                new_signature.instructions = improved_instruction
                
                # Log mutation for analysis
                self.mutation_history.append({
                    'original': current_signature.instructions,
                    'improved': improved_instruction,
                    'feedback_scores': feedback.scores,
                    'avg_score': sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
                })
                
                logger.debug(f"Mutated instruction: '{current_signature.instructions}' -> '{improved_instruction}'")
            else:
                # Fallback: make minor variation to avoid stagnation
                new_signature.instructions = self._make_variation(current_signature.instructions)
                
            return new_signature
            
        except Exception as e:
            logger.warning(f"Reflective mutation failed: {e}")
            # Fallback to original signature
            return current_signature
    
    def _create_reflection_prompt(self, signature: Signature, feedback: FeedbackResult) -> str:
        """Create analysis prompt for LLM reflection."""
        # Summarize performance
        avg_score = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
        performance_summary = f"Average score: {avg_score:.2f} on {len(feedback.scores)} examples"
        
        # Extract key failure patterns from diagnostics
        failures = [diag for diag, score in zip(feedback.diagnostics, feedback.scores) if score <= 0.5]
        failure_summary = "\n".join(failures[:3]) if failures else "No major failures observed"
        
        # Current instruction context
        current_instruction = signature.instructions or "Answer the question."
        
        # Input/output field context
        input_fields = [name for name, field in signature.fields.items() if hasattr(field, 'json_schema_extra') and field.json_schema_extra.get('__dspy_field_type') == 'input']
        output_fields = [name for name, field in signature.fields.items() if hasattr(field, 'json_schema_extra') and field.json_schema_extra.get('__dspy_field_type') == 'output']
        
        reflection_prompt = f"""Analyze this AI system's performance and suggest instruction improvements:

Current Instruction: "{current_instruction}"
Task: {' -> '.join(input_fields)} -> {' -> '.join(output_fields)}

Performance Analysis:
{performance_summary}

Failure Examples:
{failure_summary}

The instruction should be clear, specific, and guide the model toward better performance on this task. Consider what reasoning steps, constraints, or clarifications might help."""
        
        return reflection_prompt
    
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
    """Enhanced feedback with diagnostic traces."""
    
    def collect_feedback(self, program: Module, examples: List[Example], metric: Callable) -> FeedbackResult:
        """Collect enhanced feedback with DSPy traces and diagnostics.
        
        Returns FeedbackResult with:
        - traces: DSPy execution traces for each example
        - diagnostics: Human-readable diagnostic messages
        - scores: Scalar scores from metric evaluation
        """
        if not examples:
            return FeedbackResult(traces=[], diagnostics=[], scores=[])
            
        traces = []
        diagnostics = []
        scores = []
        
        for example in examples:
            try:
                # Collect trace during execution
                trace = []
                with dspy.context(trace=trace):
                    prediction = program(**example.inputs())
                    
                # Extract trace from context
                execution_trace = trace
                traces.append(execution_trace)
                
                # Compute score
                score = metric(example, prediction)
                scores.append(float(score))
                
                # Generate diagnostic message
                diagnostic = self._generate_diagnostic(example, prediction, score, execution_trace)
                diagnostics.append(diagnostic)
                
            except Exception as e:
                logger.warning(f"Failed to collect feedback for example: {e}")
                traces.append([])
                scores.append(0.0)
                diagnostics.append(f"Execution failed: {str(e)}")
                
        return FeedbackResult(traces=traces, diagnostics=diagnostics, scores=scores)
    
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
        
        This is where GEPA gets major speedup over MIPROv2 through parallel evaluation.
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
        """Step 3: Collect feedback on minibatch."""
        minibatch = create_minibatch(feedback_data, self.minibatch_size)
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
                    
            # Select two candidates from different lineages with good performance
            if len(lineage_groups) < 2:
                return None
                
            best_from_lineages = []
            for root, group in lineage_groups.items():
                # Find best candidate from this lineage
                best_idx = None
                best_score = -float('inf')
                
                for idx, lineage in group:
                    avg_score = scores.compute_average_score(idx)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_idx = idx
                        
                if best_idx is not None:
                    best_from_lineages.append((best_idx, best_score))
                    
            if len(best_from_lineages) < 2:
                return None
                
            # Sort by score and take top 2
            best_from_lineages.sort(key=lambda x: x[1], reverse=True)
            parent1_idx, parent2_idx = best_from_lineages[0][0], best_from_lineages[1][0]
            
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
    
    def _merge_candidates(self, candidate1: Module, candidate2: Module) -> Optional[Module]:
        """Merge two candidates by combining their best instructions.
        
        This is a simplified merge strategy - could be enhanced with more sophisticated
        instruction combination logic.
        """
        try:
            # Create base candidate from candidate1
            merged_candidate = candidate1.deepcopy()
            
            # Extract instructions from both candidates  
            predictors1 = candidate1.predictors()
            predictors2 = candidate2.predictors()
            
            if len(predictors1) != len(predictors2):
                return None  # Can't merge candidates with different structures
                
            # Simple merge strategy: alternate instructions or combine them
            merged_predictors = merged_candidate.predictors()
            
            for i, (pred1, pred2, merged_pred) in enumerate(zip(predictors1, predictors2, merged_predictors)):
                sig1 = get_signature(pred1)
                sig2 = get_signature(pred2)
                
                # Combine instructions (simple concatenation for now)
                inst1 = sig1.instructions or ""
                inst2 = sig2.instructions or ""
                
                if inst1 and inst2 and inst1 != inst2:
                    # Create combined instruction
                    combined_instruction = f"{inst1} Additionally, {inst2.lower()}"
                    merged_sig = get_signature(merged_pred)
                    merged_sig.instructions = combined_instruction
                    
            return merged_candidate
            
        except Exception as e:
            logger.warning(f"Candidate merge failed: {e}")
            return None
