"""Efficient Pareto Frontier selection using ScoreMatrix.

This implements the GEPA Algorithm 2 efficiently by using the ScoreMatrix
which already maintains the best candidate per task.
"""

import random
from typing import List, Set
from collections import defaultdict

from ..data.candidate import Candidate


class ParetoFrontier:
    """Selector that efficiently computes Pareto frontier from ScoreMatrix.
    
    Follows GEPA Algorithm 2 exactly:
    1. Get candidates that achieve best score on at least one task  
    2. Remove strictly dominated candidates
    3. Stochastically sample based on winning frequency
    """
    
    def __init__(self):
        self.task_winners = {}  # task_id -> candidate mapping
        self.candidate_tasks = defaultdict(list)  # candidate -> list of tasks won
    
    def append(self, task_id: int, candidate: 'Candidate') -> None:
        """Add a task winner (accumulator pattern).
        
        Args:
            task_id: The task this candidate won
            candidate: The winning candidate for this task
        """
        self.task_winners[task_id] = candidate
        self.candidate_tasks[candidate].append(task_id)
    
    def select_from_pool(self, candidate_pool) -> List[Candidate]:
        """Select Pareto frontier candidates from pool using accumulator pattern.
        
        Uses ScoreMatrix which sends task winners via accumulator pattern.
        
        Args:
            candidate_pool: CandidatePool with ScoreMatrix
            
        Returns:
            List of candidates forming the Pareto frontier
        """
        # Clear previous data
        self.task_winners.clear()
        self.candidate_tasks.clear()
        
        # Step 1: Accumulate task winners from ScoreMatrix
        candidate_pool.filter_top(self)
        
        if not self.task_winners:
            return []
            
        # Step 2: Remove dominated candidates from task winners
        unique_candidates = set(self.task_winners.values())
        pareto_candidates = self._remove_dominated_from_accumulated(unique_candidates)
        
        return pareto_candidates
    
    def select_one_stochastic(self, candidate_pool) -> Candidate:
        """Select one candidate from Pareto frontier using stochastic sampling.
        
        Candidates with higher winning frequency are more likely to be selected.
        """
        pareto_candidates = self.select_from_pool(candidate_pool)
        
        if len(pareto_candidates) <= 1:
            return pareto_candidates[0] if pareto_candidates else None
            
        # Stochastic selection based on winning frequency
        return self._stochastic_sample_from_accumulated(pareto_candidates)
    
    def generate(self, generator_strategy):
        """Generate new candidates from Pareto frontier using GeneratorStrategy.
        
        The GeneratorStrategy is responsible for creating its own output cohort.
        
        Args:
            generator_strategy: Strategy that generates candidates and manages output
            
        Returns:
            Whatever the generator_strategy returns (typically a Cohort)
        """
        # Get Pareto frontier candidates from accumulated data
        if not self.task_winners:
            return generator_strategy.create_empty_cohort()
            
        unique_candidates = set(self.task_winners.values())
        pareto_candidates = self._remove_dominated_from_accumulated(unique_candidates)
        
        # Let generator create new candidates from Pareto frontier
        return generator_strategy.generate_from_parents(pareto_candidates)
    
    def _get_task_winners(self, score_matrix) -> Set[Candidate]:
        """Get all candidates that win at least one task.
        
        This is exactly what Algorithm 2 does: only candidates that achieve
        best score on at least one task are eligible for Pareto frontier.
        
        Returns:
            Set of candidates that win at least one task
        """
        task_winners = set()
        
        # Get all task IDs that have best candidates
        all_task_ids = score_matrix.get_all_task_ids()
        
        # For each task, get the best candidate (task winner)
        for task_id in all_task_ids:
            best_candidate = score_matrix.get_best_candidate_for_task(task_id)
            if best_candidate is not None:
                task_winners.add(best_candidate)
                
        return task_winners
    
    def _remove_dominated_from_accumulated(self, candidates: Set[Candidate]) -> List[Candidate]:
        """Remove strictly dominated candidates.
        
        Candidate A dominates candidate B if:
        - A performs >= B on all tasks 
        - A performs > B on at least one task
        """
        candidates_list = list(candidates)
        non_dominated = []
        
        # Get all task IDs from accumulated data
        all_task_ids = list(self.task_winners.keys())
        
        for i, candidate_a in enumerate(candidates_list):
            is_dominated = False
            
            for j, candidate_b in enumerate(candidates_list):
                if i == j:
                    continue
                    
                # Check if candidate_b dominates candidate_a
                if self._dominates(candidate_b, candidate_a, all_task_ids):
                    is_dominated = True
                    break
                    
            if not is_dominated:
                non_dominated.append(candidate_a)
                
        return non_dominated
    
    def _dominates(self, candidate_a: Candidate, candidate_b: Candidate, task_ids: List[int]) -> bool:
        """Check if candidate_a dominates candidate_b.
        
        Returns True if A >= B on all tasks and A > B on at least one task.
        """
        dominates = True
        strictly_better_on_some = False
        
        for task_id in task_ids:
            score_a = candidate_a.task_score(task_id) or 0.0
            score_b = candidate_b.task_score(task_id) or 0.0
            
            if score_a < score_b:
                dominates = False
                break
            elif score_a > score_b:
                strictly_better_on_some = True
                
        return dominates and strictly_better_on_some
    
    def _stochastic_sample_from_accumulated(self, candidates: List[Candidate]) -> Candidate:
        """Stochastically sample based on winning frequency using accumulated data.
        
        Candidates with more task wins are more likely to be selected.
        """
        if len(candidates) == 1:
            return candidates[0]
            
        # Use accumulated candidate_tasks data for winning frequency
        candidate_wins = defaultdict(float)
        
        for candidate in candidates:
            # Each candidate gets score based on number of tasks it won
            num_tasks_won = len(self.candidate_tasks[candidate])
            candidate_wins[candidate] = float(num_tasks_won)
        
        # Convert to probabilities
        total_wins = sum(candidate_wins.values())
        if total_wins == 0:
            return random.choice(candidates)
            
        probabilities = [candidate_wins[candidate] / total_wins for candidate in candidates]
        
        # Stochastic selection
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return candidates[i]
                
        return candidates[-1]  # Fallback
    
    def _stochastic_sample(self, candidates: List[Candidate], score_matrix) -> Candidate:
        """Stochastically sample based on winning frequency.
        
        Candidates with more task wins are more likely to be selected.
        """
        if len(candidates) == 1:
            return candidates[0]
            
        # Count wins for each candidate
        candidate_wins = defaultdict(float)
        all_task_ids = score_matrix.get_all_task_ids()
        
        # For each task, find the best candidate(s) and award wins
        for task_id in all_task_ids:
            best_score = -float('inf')
            best_candidates = []
            
            for candidate in candidates:
                score = candidate.task_score(task_id) or 0.0
                if score > best_score:
                    best_score = score
                    best_candidates = [candidate]
                elif abs(score - best_score) < 1e-6:  # Tie
                    best_candidates.append(candidate)
            
            # Award fractional wins to tied candidates
            for candidate in best_candidates:
                candidate_wins[candidate] += 1.0 / len(best_candidates)
        
        # Convert to probabilities
        total_wins = sum(candidate_wins.values())
        if total_wins == 0:
            return random.choice(candidates)
            
        probabilities = [candidate_wins[candidate] / total_wins for candidate in candidates]
        
        # Stochastic selection
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return candidates[i]
                
        return candidates[-1]  # Fallback