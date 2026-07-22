"""Candidate Tree Visualization for GEPA Analysis.

This module provides tree visualization tools that match the GEPA paper's
candidate evolution diagrams, showing parent-child relationships, task
success counts, and average scores.
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass

from ..data.candidate import Candidate
from ..observers import SelectorObserver, GeneratorObserver, EvaluatorObserver

logger = logging.getLogger(__name__)


@dataclass
class CandidateNodeData:
    """Data for a single node in the candidate tree."""
    candidate: Candidate
    generation: int
    parents: List[Candidate]
    children: List[Candidate]
    avg_score: float
    task_wins: int
    task_success_count: int
    total_tasks: int
    success_rate: float
    creation_strategy: str  # "mutation", "merge", "initial"


class CandidateTreeVisualizer(SelectorObserver, GeneratorObserver, EvaluatorObserver):
    """Visualize candidate evolution trees as shown in GEPA paper.
    
    This class tracks all candidates created during optimization and
    provides various visualization methods to understand the evolutionary
    process, including ASCII trees, statistics, and debug information.
    """
    
    def __init__(self, track_detailed_stats: bool = True):
        """Initialize the tree visualizer.
        
        Args:
            track_detailed_stats: Whether to track detailed per-task statistics
        """
        self.candidates: Dict[int, CandidateNodeData] = {}  # candidate_id -> node_data
        self.generation_candidates: Dict[int, List[int]] = defaultdict(list)  # generation -> candidate_ids
        self.parent_child_map: Dict[int, List[int]] = defaultdict(list)  # parent_id -> child_ids
        self.root_candidates: Set[int] = set()  # Candidates with no parents
        self.cohort_snapshots: List[Dict[str, Any]] = []
        
        self.track_detailed_stats = track_detailed_stats
        self.total_tasks = 0
        
        # Statistics
        self.stats = {
            "total_candidates": 0,
            "generations": 0,
            "mutation_count": 0,
            "merge_count": 0,
            "success_lineages": 0
        }
    
    def set_total_tasks(self, total_tasks: int) -> None:
        """Set the total number of tasks for success rate calculations."""
        self.total_tasks = total_tasks
    
    def add_candidate(self, 
                     candidate: Candidate, 
                     task_wins: int = 0,
                     task_success_count: int = 0,
                     creation_strategy: str = "unknown") -> None:
        """Add a candidate to the tree visualization.
        
        Args:
            candidate: The candidate to add
            task_wins: Number of tasks this candidate wins (best score on)
            task_success_count: Number of tasks this candidate succeeds on (score > threshold)
            creation_strategy: How this candidate was created ("mutation", "merge", "initial")
        """
        candidate_id = id(candidate)
        
        # Skip if already tracked
        if candidate_id in self.candidates:
            return
        
        # Calculate metrics
        avg_score = candidate.average_score()
        success_rate = task_success_count / self.total_tasks if self.total_tasks > 0 else 0.0
        
        # Create node data
        node_data = CandidateNodeData(
            candidate=candidate,
            generation=candidate.generation_number,
            parents=candidate.parents.copy(),
            children=[],  # Will be populated as children are added
            avg_score=avg_score,
            task_wins=task_wins,
            task_success_count=task_success_count,
            total_tasks=self.total_tasks,
            success_rate=success_rate,
            creation_strategy=creation_strategy
        )
        
        # Store in tracking structures
        self.candidates[candidate_id] = node_data
        self.generation_candidates[candidate.generation_number].append(candidate_id)
        
        # Update parent-child relationships
        if candidate.parents:
            for parent in candidate.parents:
                parent_id = id(parent)
                self.parent_child_map[parent_id].append(candidate_id)
                
                # Add to parent's children list if parent is tracked
                if parent_id in self.candidates:
                    self.candidates[parent_id].children.append(candidate)
        else:
            self.root_candidates.add(candidate_id)
        
        # Update statistics
        self.stats["total_candidates"] += 1
        self.stats["generations"] = max(self.stats["generations"], candidate.generation_number)
        
        if creation_strategy == "mutation":
            self.stats["mutation_count"] += 1
        elif creation_strategy == "merge":
            self.stats["merge_count"] += 1
    
    def render_ascii_tree(self, show_scores: bool = True, show_task_wins: bool = True) -> str:
        """Render an ASCII tree showing candidate relationships.
        
        Args:
            show_scores: Whether to show average scores
            show_task_wins: Whether to show task win counts
            
        Returns:
            String containing the ASCII tree representation
        """
        if not self.candidates:
            return "No candidates to display"
        
        output = []
        output.append("Candidate Evolution Tree")
        output.append("=" * 50)
        
        # Group by generation and display
        for generation in sorted(self.generation_candidates.keys()):
            candidate_ids = self.generation_candidates[generation]
            
            output.append(f"\nGeneration {generation} ({len(candidate_ids)} candidates):")
            
            for i, candidate_id in enumerate(candidate_ids):
                node_data = self.candidates[candidate_id]
                
                # Tree connector
                connector = "├─" if i < len(candidate_ids) - 1 else "└─"
                
                # Build node label
                label_parts = [f"Candidate(gen={generation}"]
                
                if show_scores:
                    label_parts.append(f"μ={node_data.avg_score:.3f}")
                
                if show_task_wins:
                    label_parts.append(f"wins={node_data.task_wins}")
                
                if node_data.creation_strategy != "unknown":
                    label_parts.append(f"via={node_data.creation_strategy}")
                
                label = ", ".join(label_parts) + ")"
                
                output.append(f"  {connector} {label}")
                
                # Show parent relationships
                if node_data.parents:
                    parent_gens = [p.generation_number for p in node_data.parents]
                    parent_info = f"parents=gen{parent_gens}"
                    output.append(f"      └─ {parent_info}")
        
        return "\\n".join(output)

    def record_cohort(self, cohort, label: str | None = None, step: int | None = None) -> None:
        """Record a cohort membership snapshot for optional DOT export."""
        self.cohort_snapshots.append({
            "label": label or type(cohort).__name__,
            "step": step if step is not None else len(self.cohort_snapshots),
            "type": f"{type(cohort).__module__}:{type(cohort).__qualname__}",
            "candidate_ids": [id(candidate) for candidate in cohort],
        })

    def render_dot(self, show_scores: bool = True, include_cohorts: bool = False) -> str:
        """Render the tracked candidate lineage as Graphviz DOT.

        The output is dependency-free; callers may pass it to Graphviz or
        display it with any DOT-compatible renderer.
        """
        lines = ["digraph DarwinCandidates {", "  rankdir=LR;"]
        if not self.candidates:
            lines.append("}")
            return "\n".join(lines)

        def quote(value: str) -> str:
            return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ') + '"'

        for candidate_id, node in self.candidates.items():
            label = f"gen={node.generation}"
            if show_scores:
                label += f"\\nscore={node.avg_score:.3f}"
            label += f"\\nvia={node.creation_strategy}"
            lines.append(f"  c{candidate_id} [label={quote(label)}];")

        for candidate_id, node in self.candidates.items():
            for parent in node.parents:
                parent_id = id(parent)
                if parent_id in self.candidates:
                    lines.append(f"  c{parent_id} -> c{candidate_id};")
        if include_cohorts:
            for index, snapshot in enumerate(self.cohort_snapshots):
                cluster_id = f"cohort_{index}"
                label = f"{snapshot['label']} (step {snapshot['step']})"
                lines.append(f"  subgraph cluster_{cluster_id} {{")
                lines.append(f"    label={quote(label)};")
                lines.append("    color=gray;")
                cohort_node = f"cohort_node_{index}"
                lines.append(f"    {cohort_node} [shape=box, label={quote(snapshot['type'])}];")
                for candidate_id in snapshot["candidate_ids"]:
                    if candidate_id in self.candidates:
                        lines.append(
                            f"    {cohort_node} -> c{candidate_id} "
                            "[style=dashed, color=gray];"
                        )
                lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    to_dot = render_dot
    
    def render_detailed_tree(self) -> str:
        """Render a detailed tree with full statistics."""
        if not self.candidates:
            return "No candidates to display"
        
        output = []
        output.append("Detailed Candidate Evolution Tree")
        output.append("=" * 60)
        
        # Show overall statistics first
        output.append(f"\nOverall Statistics:")
        output.append(f"  Total candidates: {self.stats['total_candidates']}")
        output.append(f"  Generations: {self.stats['generations'] + 1}")
        output.append(f"  Mutations: {self.stats['mutation_count']}")
        output.append(f"  Merges: {self.stats['merge_count']}")
        output.append(f"  Total tasks: {self.total_tasks}")
        
        # Show tree by generation
        for generation in sorted(self.generation_candidates.keys()):
            candidate_ids = self.generation_candidates[generation]
            
            output.append(f"\n{'='*20} Generation {generation} {'='*20}")
            
            for candidate_id in candidate_ids:
                node_data = self.candidates[candidate_id]
                
                output.append(f"\nCandidate {candidate_id}:")
                output.append(f"  Strategy: {node_data.creation_strategy}")
                output.append(f"  Average Score: {node_data.avg_score:.3f}")
                output.append(f"  Task Wins: {node_data.task_wins}")
                output.append(f"  Success Count: {node_data.task_success_count}/{node_data.total_tasks}")
                output.append(f"  Success Rate: {node_data.success_rate:.1%}")
                
                if node_data.parents:
                    parent_info = ", ".join([f"gen{p.generation_number}" for p in node_data.parents])
                    output.append(f"  Parents: {parent_info}")
                
                if node_data.children:
                    child_info = ", ".join([f"gen{c.generation_number}" for c in node_data.children])
                    output.append(f"  Children: {child_info}")
        
        return "\\n".join(output)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the candidate tree."""
        if not self.candidates:
            return {"error": "No candidates tracked"}
        
        # Basic stats
        stats = self.stats.copy()
        
        # Calculate generation-wise statistics
        gen_stats = {}
        for generation, candidate_ids in self.generation_candidates.items():
            gen_data = [self.candidates[cid] for cid in candidate_ids]
            
            gen_stats[generation] = {
                "count": len(gen_data),
                "avg_score": sum(node.avg_score for node in gen_data) / len(gen_data),
                "max_score": max(node.avg_score for node in gen_data),
                "min_score": min(node.avg_score for node in gen_data),
                "total_task_wins": sum(node.task_wins for node in gen_data),
                "avg_success_rate": sum(node.success_rate for node in gen_data) / len(gen_data)
            }
        
        stats["generation_stats"] = gen_stats
        
        # Strategy effectiveness
        if self.stats["mutation_count"] > 0 or self.stats["merge_count"] > 0:
            mutation_nodes = [node for node in self.candidates.values() if node.creation_strategy == "mutation"]
            merge_nodes = [node for node in self.candidates.values() if node.creation_strategy == "merge"]
            
            stats["strategy_effectiveness"] = {
                "mutation_avg_score": sum(node.avg_score for node in mutation_nodes) / len(mutation_nodes) if mutation_nodes else 0,
                "merge_avg_score": sum(node.avg_score for node in merge_nodes) / len(merge_nodes) if merge_nodes else 0,
                "mutation_success_rate": sum(node.success_rate for node in mutation_nodes) / len(mutation_nodes) if mutation_nodes else 0,
                "merge_success_rate": sum(node.success_rate for node in merge_nodes) / len(merge_nodes) if merge_nodes else 0
            }
        
        # Lineage analysis
        successful_lineages = self._analyze_successful_lineages()
        stats["successful_lineages"] = len(successful_lineages)
        stats["lineage_analysis"] = successful_lineages
        
        return stats
    
    def export_json(self) -> str:
        """Export tree data as JSON for external analysis."""
        export_data = {
            "metadata": {
                "total_candidates": self.stats["total_candidates"],
                "generations": self.stats["generations"],
                "total_tasks": self.total_tasks
            },
            "candidates": {},
            "relationships": {
                "parent_child": dict(self.parent_child_map),
                "generation_groups": dict(self.generation_candidates)
            }
        }
        
        # Export candidate data
        for candidate_id, node_data in self.candidates.items():
            export_data["candidates"][str(candidate_id)] = {
                "generation": node_data.generation,
                "avg_score": node_data.avg_score,
                "task_wins": node_data.task_wins,
                "task_success_count": node_data.task_success_count,
                "success_rate": node_data.success_rate,
                "creation_strategy": node_data.creation_strategy,
                "parent_ids": [id(p) for p in node_data.parents],
                "child_ids": [id(c) for c in node_data.children]
            }
        
        return json.dumps(export_data, indent=2)
    
    def find_best_lineage(self) -> Optional[List[Candidate]]:
        """Find the lineage that led to the best performing candidate."""
        if not self.candidates:
            return None
        
        # Find best candidate
        best_node = max(self.candidates.values(), key=lambda node: node.avg_score)
        
        # Trace back to root
        lineage = []
        current = best_node.candidate
        
        while current:
            lineage.append(current)
            if current.parents:
                # For simplicity, follow first parent if multiple
                current = current.parents[0]
            else:
                break
        
        lineage.reverse()  # Root to best
        return lineage
    
    def _analyze_successful_lineages(self, success_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Analyze lineages that achieved high success rates."""
        successful_lineages = []
        
        # Find candidates above threshold
        successful_candidates = [
            node for node in self.candidates.values() 
            if node.success_rate >= success_threshold
        ]
        
        for node in successful_candidates:
            # Trace lineage back to root
            lineage = []
            current = node.candidate
            
            while current:
                lineage.append(current)
                if current.parents:
                    current = current.parents[0]  # Follow first parent
                else:
                    break
            
            lineage.reverse()
            
            lineage_info = {
                "final_score": node.avg_score,
                "success_rate": node.success_rate,
                "length": len(lineage),
                "generations": [c.generation_number for c in lineage],
                "strategies": []  # Would need to track creation strategies
            }
            
            successful_lineages.append(lineage_info)
        
        return successful_lineages
    
    def reset(self) -> None:
        """Reset the visualizer state."""
        self.candidates.clear()
        self.generation_candidates.clear()
        self.parent_child_map.clear()
        self.root_candidates.clear()
        self.cohort_snapshots.clear()
        self.stats = {
            "total_candidates": 0,
            "generations": 0,
            "mutation_count": 0,
            "merge_count": 0,
            "success_lineages": 0
        }
    
    # Observer protocol implementations
    async def promote(self, survivors) -> None:
        """Observe candidate promotion from selector."""
        self.record_cohort(survivors, label="survivors", step=survivors.iteration)
        for candidate in survivors.candidates:
            # Update existing candidate with promotion info
            candidate_id = id(candidate)
            if candidate_id in self.candidates:
                node_data = self.candidates[candidate_id]
                # Could update promotion-specific data here
    
    async def update_score(self, candidate: Candidate, task_id: int, score: float) -> None:
        """Observe single score update from selector."""
        # Update task wins and success counts
        candidate_id = id(candidate)
        if candidate_id in self.candidates:
            # Recalculate metrics when scores change
            self._update_candidate_metrics(candidate)
    
    async def update_score_batch(self, scores: Dict) -> None:
        """Observe batch score updates from selector."""
        for candidate in scores.keys():
            self._update_candidate_metrics(candidate)
    
    async def generate(self, parents, newborns) -> None:
        """Observe candidate generation from generator."""
        self.record_cohort(parents, label="parents", step=parents.iteration)
        self.record_cohort(newborns, label="newborns", step=newborns.iteration)
        for candidate in newborns.candidates:
            # Determine creation strategy based on parents
            creation_strategy = "unknown"
            if len(candidate.parents) == 0:
                creation_strategy = "initial"
            elif len(candidate.parents) == 1:
                creation_strategy = "mutation"
            elif len(candidate.parents) > 1:
                creation_strategy = "merge"
            
            self.add_candidate(candidate, creation_strategy=creation_strategy)
    
    async def filter_candidates(self, candidates: List[Candidate]) -> None:
        """Observe candidate filtering from generator."""
        # Could track filtered candidates for analysis
        pass
    
    async def update_instruction(self, old_instruction: str, new_instruction: str, candidate: Candidate) -> None:
        """Observe instruction updates from generator."""
        # Could track instruction evolution
        pass
    
    async def evaluate(self, candidates, results) -> None:
        """Observe candidate evaluation from evaluator."""
        for candidate in results.candidates:
            self._update_candidate_metrics(candidate)
    
    def _update_candidate_metrics(self, candidate: Candidate) -> None:
        """Update metrics for a candidate based on current scores."""
        candidate_id = id(candidate)
        if candidate_id in self.candidates:
            node_data = self.candidates[candidate_id]
            # Recalculate average score
            node_data.avg_score = candidate.average_score()
            # Recalculate success count (assuming 0.5 threshold)
            node_data.task_success_count = sum(1 for score in candidate.scores if float(score.value) > 0.5)
            node_data.success_rate = node_data.task_success_count / self.total_tasks if self.total_tasks > 0 else 0.0
