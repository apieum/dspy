"""Core GEPA implementation - main optimization class."""

import asyncio
import logging
import statistics
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy
from dspy.teleprompt.bootstrap_finetune import FinetuneTeleprompter
from dspy.teleprompt.utils import create_minibatch, get_signature, set_signature

from .budget import BudgetTracker, CombinedBudget, LLMCallsBudget
from .data import (
    CandidateLineage,
    CandidatePool,
    FeedbackResult,
    ScoreMatrix,
    SplitDataset,
    TrainingDataset,
)
from .feedback import EnhancedFeedbackCollector, ReflectivePromptMutator, RoundRobinModuleSelector
from .generators import CompositeGenerator, CrossoverGenerator, MutationGenerator
from .selection import ParetoCandidateSelector

logger = logging.getLogger(__name__)


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
        max_errors: Optional[int] = None,
        num_threads: Optional[int] = None,
        candidate_generator=None,
        candidate_selector=None,
        feedback_collector=None,
        pareto_ratio: float = 0.67,
        max_candidates: int = 50,
        prompt_model=None,
    ):
        super().__init__()
        self.metric = metric
        self.minibatch_size = minibatch_size
        self.max_errors = max_errors
        self.num_threads = num_threads
        self.pareto_ratio = pareto_ratio
        self.max_candidates = max_candidates
        self.prompt_model = prompt_model

        # New architecture components with defaults
        self.candidate_generator = candidate_generator or CompositeGenerator([
            MutationGenerator(
                prompt_mutator=ReflectivePromptMutator(),
                module_selector=RoundRobinModuleSelector(),
                feedback_collector=EnhancedFeedbackCollector()
            ),
            CrossoverGenerator(frequency=5, min_candidates=3)
        ])
        self.candidate_selector = candidate_selector or ParetoCandidateSelector()
        self.feedback_collector = feedback_collector or EnhancedFeedbackCollector()

    def compile(
        self,
        student: dspy.Module,
        *,
        dataset: Optional[TrainingDataset] = None,
        trainset: Optional[List[dspy.Example]] = None,  # Legacy support
        teacher: Optional[dspy.Module] = None,
        valset: Optional[List[dspy.Example]] = None,
        **kwargs
    ) -> dspy.Module:
        """
        Main GEPA compilation implementing Algorithm 1 from paper.

        Args:
            student: Program to optimize
            dataset: Training dataset following TrainingDataset protocol
            trainset: Legacy list of examples (for compatibility)
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

        # 2. Handle dataset input (new protocol or legacy)
        if dataset is not None:
            training_dataset = dataset
        elif trainset is not None:
            # Convert legacy trainset to SplitDataset
            training_dataset = SplitDataset(trainset, pareto_ratio=0.67)
        else:
            raise ValueError("Either 'dataset' or 'trainset' must be provided")

        # 3. Initialize new architecture components
        candidate_pool = CandidatePool()
        
        # Handle budget parameter - can be an object or legacy int
        budget_param = kwargs.get('budget')
        if isinstance(budget_param, BudgetTracker):
            budget = budget_param
        elif isinstance(budget_param, int):
            # Legacy support: int budget = LLM calls budget
            max_iterations = kwargs.get('max_iterations')
            if max_iterations is not None:
                budget = CombinedBudget(llm_limit=budget_param, max_iterations=max_iterations)
            else:
                budget = LLMCallsBudget(limit=budget_param)
        else:
            # Default budget
            default_limit = len(list(training_dataset.feedback_data())) * 10
            budget = LLMCallsBudget(limit=default_limit)

        # 4. Create initial candidate and add to pool
        initial_program = student.deepcopy()
        initial_lineage = CandidateLineage(
            candidate_id=0, 
            parent_id=None, 
            generation=0, 
            mutation_type="initial", 
            creation_iteration=0
        )
        candidate_pool.add_candidate(initial_program, initial_lineage)

        logger.info(f"Initialized candidate pool with 1 candidate")

        # 5. Initial Pareto evaluation
        logger.info("Performing initial Pareto evaluation...")
        pareto_data = list(training_dataset.pareto_data())
        self._evaluate_candidates_on_pareto(candidate_pool, pareto_data, budget)

        # 6. Main optimization loop (Algorithm 1)
        iteration = 0
        feedback_data = list(training_dataset.feedback_data())
        
        while budget.has_budget() and candidate_pool.size() > 0:
            iteration += 1
            budget.warn_iteration_start(iteration)
            budget.add_iteration_cost(1)  # Beginning of iteration cost
            logger.info(f"GEPA iteration {iteration}, budget: {budget.get_stats()}")
            
            # Check budget again after iteration cost
            if not budget.has_budget():
                logger.info(f"Budget exhausted after iteration cost")
                break

            try:
                # Generate new candidates using strategy
                new_candidates = self.candidate_generator.generate_candidates(
                    candidate_pool, feedback_data, iteration
                )
                
                if not new_candidates:
                    logger.info("No new candidates generated, consuming minimal budget and continuing...")
                    # Consume minimal budget even when generation fails to prevent infinite loops
                    budget.add_minibatch_cost(1)
                    continue

                logger.info(f"Generated {len(new_candidates)} new candidates")

                # Evaluate new candidates on Pareto set
                for candidate in new_candidates:
                    if not budget.has_budget():
                        break
                    
                    # Add to pool temporarily for evaluation
                    candidate_id = candidate_pool.add_candidate(candidate)
                    
                    # Evaluate on Pareto set
                    self._evaluate_single_candidate(candidate_pool, candidate_id, pareto_data, budget)

                # Optionally trim candidate pool if it gets too large
                if candidate_pool.size() > 20:  # Reasonable limit
                    self._trim_candidate_pool(candidate_pool)

            except Exception as e:
                logger.warning(f"Iteration {iteration} failed: {e}")
                continue

        # 7. Select best candidate
        best_candidate = self._select_best_candidate(candidate_pool)
        
        # 8. Mark as compiled and return
        best_candidate._compiled = True
        logger.info(f"GEPA compilation complete after {iteration} iterations")
        logger.info(f"Final budget usage: {budget.get_stats()}")
        
        return best_candidate

    def _split_dataset(self, trainset: List[dspy.Example]) -> Tuple[List[dspy.Example], List[dspy.Example]]:
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
        candidate_pool: CandidatePool,
        pareto_data: List[dspy.Example], 
        budget: BudgetTracker
    ):
        """Evaluate candidates on Pareto set using visitor pattern.
        """
        if not pareto_data or not candidate_pool.candidates:
            return

        # Define evaluation function using closure to capture context
        def evaluate_candidate_on_task(candidate, task_idx, example):
            """Evaluation function for visitor pattern."""
            with dspy.context(lm=self.prompt_model):
                prediction = candidate(**example.inputs())
            return self.metric(example, prediction, trace=None)

        # Use visitor pattern - much cleaner!
        evaluations_performed = candidate_pool.evaluate(
            evaluation_function=evaluate_candidate_on_task,
            tasks=pareto_data,
            skip_evaluated=True
        )

        # Update budget - count evaluations
        if evaluations_performed > 0:
            budget.add_pareto_cost(evaluations_performed)
            logger.debug(f"Pareto evaluation completed: {evaluations_performed} evaluations counted in budget")

    async def _evaluate_candidates_async(
        self,
        candidate_pool: CandidatePool,
        pareto_data: List[dspy.Example],
        budget: BudgetTracker
    ):
        """Async implementation of Pareto evaluation."""
        evaluation_tasks = []

        # Create tasks for all candidate-example pairs
        for candidate_id, candidate in enumerate(candidate_pool.candidates):
            # Skip if already evaluated
            candidate_scores = candidate_pool.get_candidate_scores(candidate_id)
            if len(candidate_scores) >= len(pareto_data):
                continue

            for task_idx, example in enumerate(pareto_data):
                # Skip if already evaluated
                if candidate_pool.get_score(candidate_id, task_idx) is not None:
                    continue

                task = self._evaluate_single_candidate_example(
                    candidate, example, candidate_id, task_idx
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

                candidate_id, task_idx, score = result
                candidate_pool.set_score(candidate_id, task_idx, score)

            # Update budget - Pareto evaluations ARE expensive LLM calls that count
            successful_evals = len([r for r in results if not isinstance(r, Exception)])
            budget.add_pareto_cost(successful_evals)
            logger.debug(f"Pareto evaluation completed: {successful_evals} evaluations counted in budget")

    def _evaluate_single_candidate(
        self,
        candidate_pool: CandidatePool,
        candidate_id: int,
        pareto_data: List[dspy.Example],
        budget: BudgetTracker
    ):
        """Evaluate a single candidate on all Pareto examples using visitor pattern."""
        if candidate_id >= len(candidate_pool.candidates):
            logger.warning(f"Candidate {candidate_id} not found in pool")
            return

        # Define evaluation function using closure to capture context
        def evaluate_candidate_on_task(candidate, task_idx, example):
            """Evaluation function for visitor pattern."""
            with dspy.context(lm=self.prompt_model):
                prediction = candidate(**example.inputs())
            return self.metric(example, prediction, trace=None)

        # Use visitor pattern for single candidate evaluation - much cleaner!
        evaluations_performed = candidate_pool.evaluate_specific_candidates(
            evaluation_function=evaluate_candidate_on_task,
            tasks=pareto_data,
            candidate_ids=[candidate_id],
            skip_evaluated=True
        )

        # Count as budget usage
        if evaluations_performed > 0:
            budget.add_pareto_cost(evaluations_performed)

    def _trim_candidate_pool(self, candidate_pool: CandidatePool):
        """Trim candidate pool to maximum size, keeping best performers."""
        if len(candidate_pool.candidates) <= self.max_candidates:
            return
        
        # Calculate average scores for all candidates
        candidate_scores = []
        for candidate_id in range(len(candidate_pool.candidates)):
            avg_score = candidate_pool.compute_average_score(candidate_id)
            candidate_scores.append((candidate_id, avg_score))
        
        # Sort by average score (descending)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the best candidates
        candidates_to_keep = set(cid for cid, _ in candidate_scores[:self.max_candidates])
        candidates_to_remove = [cid for cid in range(len(candidate_pool.candidates)) if cid not in candidates_to_keep]
        
        # Remove worst candidates from the end to avoid index shifting
        for candidate_id in sorted(candidates_to_remove, reverse=True):
            candidate_pool.remove_candidate(candidate_id)
        
        logger.info(f"Trimmed candidate pool from {len(candidate_pool.candidates) + len(candidates_to_remove)} to {len(candidate_pool.candidates)} candidates")

    async def _evaluate_single_candidate_example(
        self,
        candidate: dspy.Module,
        example: dspy.Example,
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

    def _select_candidate_step(self, candidates: List[dspy.Module], scores: ScoreMatrix) -> int:
        """Step 1: Select candidate using Pareto-based strategy."""
        return self.candidate_selector.select_candidate(candidates, scores)

    def _select_module_step(self, candidate: dspy.Module) -> int:
        """Step 2: Select module to mutate."""
        return self.module_selector.select_module(candidate)

    def _collect_feedback_step(
        self,
        candidate: dspy.Module,
        feedback_data: List[dspy.Example],
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
        candidate: dspy.Module,
        module_idx: int,
        feedback: FeedbackResult,
        budget: BudgetTracker
    ) -> dspy.Module:
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
        new_candidate: dspy.Module,
        parent_candidate: dspy.Module,
        feedback_data: List[dspy.Example],
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

    def _evaluate_candidate_on_batch(self, candidate: dspy.Module, batch: List[dspy.Example]) -> float:
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

    def _select_best_candidate(self, candidate_pool: CandidatePool) -> dspy.Module:
        """Select best candidate based on average Pareto scores.

        Returns the candidate with highest average score across Pareto evaluation set.
        """
        if not candidate_pool.candidates:
            raise ValueError("No candidates to select from")

        if len(candidate_pool.candidates) == 1:
            return candidate_pool.candidates[0]

        best_candidate = None
        best_candidate_id = None
        best_score = -float('inf')

        for candidate_id, candidate in enumerate(candidate_pool.candidates):
            avg_score = candidate_pool.compute_average_score(candidate_id)

            if avg_score > best_score:
                best_score = avg_score
                best_candidate = candidate
                best_candidate_id = candidate_id

        # Log ancestry information for best candidate
        best_lineage = candidate_pool.get_lineage(best_candidate_id)
        if best_lineage:
            logger.info(f"Selected best candidate: generation {best_lineage.generation}, mutation_type: {best_lineage.mutation_type}")
            logger.info(f"Best candidate average score: {best_score:.3f}")
        else:
            logger.info(f"Selected best candidate with average score: {best_score:.3f}")
        return best_candidate

    def _register_candidate(self, candidate: dspy.Module, parent_id: Optional[int], mutation_type: str, iteration: int):
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

    def _attempt_merge(self, candidates: List[dspy.Module], scores: ScoreMatrix, iteration: int) -> Optional[dspy.Module]:
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
                             candidates: List[dspy.Module]) -> Tuple[int, int]:
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

    def _calculate_instruction_compatibility(self, candidate1: dspy.Module, candidate2: dspy.Module) -> float:
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

    def _merge_candidates(self, candidate1: dspy.Module, candidate2: dspy.Module) -> Optional[dspy.Module]:
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