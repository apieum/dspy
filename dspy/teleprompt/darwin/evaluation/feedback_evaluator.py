"""
Feedback Evaluation Framework for Darwin GEPA

This module provides metrics and evaluation tools for assessing the quality
of abstract feedback and its impact on instruction evolution.
"""

import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

import dspy
from .feedback import FeedbackResult

logger = logging.getLogger(__name__)


@dataclass
class FeedbackQualityMetrics:
    """Metrics for evaluating feedback quality."""
    abstraction_score: float  # How well feedback avoids task-specific content
    informativeness_score: float  # How much actionable information feedback provides
    generalization_score: float  # How well feedback enables general instruction improvement
    instruction_improvement_score: float  # How much instructions actually improve
    overfitting_penalty: float  # Penalty for task-specific instruction generation
    overall_quality: float  # Combined quality score


class FeedbackEvaluator:
    """Evaluates the quality of feedback and resulting instruction evolution."""
    
    def __init__(self):
        self.task_specific_terms = [
            # Scientific terms
            'heisenberg', 'uncertainty', 'principle', 'dna', 'photosynthesis', 'mitochondria',
            # Mathematical terms  
            'fibonacci', 'pythagorean', 'euclidean', 'newton', 'calculus',
            # Historical terms
            'napoleon', 'caesar', 'shakespeare', 'beethoven', 'renaissance',
            # Geographical terms
            'paris', 'london', 'tokyo', 'amazon', 'sahara', 'everest',
            # Common overfitting indicators
            'explain the', 'describe the', 'what is the'
        ]
    
    def evaluate_feedback_quality(self, 
                                  original_feedback: str, 
                                  abstract_feedback: str,
                                  original_instruction: str,
                                  evolved_instruction: str,
                                  performance_improvement: float) -> FeedbackQualityMetrics:
        """Evaluate the quality of abstract feedback and instruction evolution."""
        
        # 1. Abstraction Quality: How well does feedback avoid specific content?
        abstraction_score = self._evaluate_abstraction(original_feedback, abstract_feedback)
        
        # 2. Informativeness: How much actionable information does feedback provide?
        informativeness_score = self._evaluate_informativeness(abstract_feedback)
        
        # 3. Generalization: How general-purpose is the evolved instruction?
        generalization_score = self._evaluate_generalization(evolved_instruction)
        
        # 4. Instruction Improvement: How much did the instruction actually improve?
        improvement_score = max(0.0, min(1.0, performance_improvement))
        
        # 5. Overfitting Penalty: Penalty for task-specific instructions
        overfitting_penalty = self._calculate_overfitting_penalty(evolved_instruction)
        
        # 6. Overall Quality: Weighted combination
        overall_quality = (
            0.25 * abstraction_score +
            0.25 * informativeness_score +
            0.30 * generalization_score +
            0.20 * improvement_score -
            0.10 * overfitting_penalty
        )
        overall_quality = max(0.0, min(1.0, overall_quality))
        
        return FeedbackQualityMetrics(
            abstraction_score=abstraction_score,
            informativeness_score=informativeness_score,
            generalization_score=generalization_score,
            instruction_improvement_score=improvement_score,
            overfitting_penalty=overfitting_penalty,
            overall_quality=overall_quality
        )
    
    def _evaluate_abstraction(self, original_feedback: str, abstract_feedback: str) -> float:
        """Evaluate how well feedback abstracts away task-specific content."""
        if not original_feedback or not abstract_feedback:
            return 0.0
        
        # Count task-specific terms in original vs abstract feedback
        original_specific_count = sum(1 for term in self.task_specific_terms 
                                     if term.lower() in original_feedback.lower())
        abstract_specific_count = sum(1 for term in self.task_specific_terms 
                                     if term.lower() in abstract_feedback.lower())
        
        # Check for structural patterns indicating good abstraction
        abstract_patterns = [
            'schema:', 'structure:', 'response:', 'analysis:', 'performance:',
            'case ', 'status:', 'issue:', 'completeness', 'reasoning'
        ]
        pattern_count = sum(1 for pattern in abstract_patterns 
                           if pattern.lower() in abstract_feedback.lower())
        
        # Score based on term reduction and structural patterns
        if original_specific_count == 0:
            abstraction_base = 0.8  # No specific terms to abstract
        else:
            reduction_ratio = max(0, (original_specific_count - abstract_specific_count) / original_specific_count)
            abstraction_base = reduction_ratio
        
        # Bonus for structural patterns
        pattern_bonus = min(0.2, pattern_count * 0.05)
        
        return min(1.0, abstraction_base + pattern_bonus)
    
    def _evaluate_informativeness(self, abstract_feedback: str) -> float:
        """Evaluate how much actionable information the feedback provides."""
        if not abstract_feedback:
            return 0.0
        
        # Look for informative elements
        informative_indicators = [
            'score', 'performance', 'issue', 'analysis', 'structure',
            'completeness', 'reasoning', 'quality', 'length', 'schema'
        ]
        
        indicator_count = sum(1 for indicator in informative_indicators
                             if indicator.lower() in abstract_feedback.lower())
        
        # Look for quantitative information
        numeric_patterns = re.findall(r'\d+\.?\d*', abstract_feedback)
        has_metrics = len(numeric_patterns) > 0
        
        # Look for structured format
        has_structure = any(marker in abstract_feedback for marker in [':', '→', 'Case', 'Status'])
        
        # Combine indicators
        base_score = min(0.7, indicator_count * 0.1)
        metric_bonus = 0.15 if has_metrics else 0.0
        structure_bonus = 0.15 if has_structure else 0.0
        
        return min(1.0, base_score + metric_bonus + structure_bonus)
    
    def _evaluate_generalization(self, evolved_instruction: str) -> float:
        """Evaluate how general-purpose the evolved instruction is."""
        if not evolved_instruction:
            return 0.0
        
        # Penalty for task-specific terms
        specific_term_count = sum(1 for term in self.task_specific_terms
                                 if term.lower() in evolved_instruction.lower())
        
        # Look for general instruction quality indicators
        quality_indicators = [
            'clear', 'accurate', 'comprehensive', 'detailed', 'structured',
            'reasoning', 'step-by-step', 'systematic', 'thorough', 'precise'
        ]
        
        quality_count = sum(1 for indicator in quality_indicators
                           if indicator.lower() in evolved_instruction.lower())
        
        # Look for general instruction patterns
        general_patterns = [
            'answer questions', 'provide', 'ensure', 'use', 'include',
            'explain', 'analyze', 'consider', 'demonstrate'
        ]
        
        pattern_count = sum(1 for pattern in general_patterns
                           if pattern.lower() in evolved_instruction.lower())
        
        # Calculate score
        base_score = 0.6  # Start with reasonable base
        quality_bonus = min(0.3, quality_count * 0.05)
        pattern_bonus = min(0.2, pattern_count * 0.03)
        specificity_penalty = min(0.5, specific_term_count * 0.1)
        
        return max(0.0, base_score + quality_bonus + pattern_bonus - specificity_penalty)
    
    def _calculate_overfitting_penalty(self, evolved_instruction: str) -> float:
        """Calculate penalty for task-specific overfitting in instructions."""
        if not evolved_instruction:
            return 0.0
        
        # Count task-specific terms
        specific_count = sum(1 for term in self.task_specific_terms
                            if term.lower() in evolved_instruction.lower())
        
        # Look for overfitting patterns
        overfitting_patterns = [
            r'explain the \w+', r'describe the \w+', r'what is the \w+',
            r'principle', r'theorem', r'law of', r'definition of'
        ]
        
        pattern_matches = sum(1 for pattern in overfitting_patterns
                             if re.search(pattern, evolved_instruction.lower()))
        
        # Calculate penalty
        term_penalty = min(0.5, specific_count * 0.1)
        pattern_penalty = min(0.3, pattern_matches * 0.1)
        
        return term_penalty + pattern_penalty

    def evaluate_feedback_evolution(self, 
                                   feedback_history: List[Tuple[str, str, str, float]]) -> Dict[str, Any]:
        """Evaluate feedback quality evolution over multiple iterations.
        
        Args:
            feedback_history: List of (original_feedback, abstract_feedback, evolved_instruction, performance)
        
        Returns:
            Dictionary with evolution metrics
        """
        if not feedback_history:
            return {'error': 'No feedback history provided'}
        
        iteration_metrics = []
        for i, (orig_fb, abs_fb, instruction, performance) in enumerate(feedback_history):
            prev_instruction = feedback_history[i-1][2] if i > 0 else "Base instruction"
            metrics = self.evaluate_feedback_quality(
                orig_fb, abs_fb, prev_instruction, instruction, performance
            )
            iteration_metrics.append(metrics)
        
        # Calculate evolution trends
        abstraction_trend = self._calculate_trend([m.abstraction_score for m in iteration_metrics])
        informativeness_trend = self._calculate_trend([m.informativeness_score for m in iteration_metrics])
        generalization_trend = self._calculate_trend([m.generalization_score for m in iteration_metrics])
        
        # Overall system performance
        avg_quality = sum(m.overall_quality for m in iteration_metrics) / len(iteration_metrics)
        final_quality = iteration_metrics[-1].overall_quality if iteration_metrics else 0.0
        
        return {
            'iterations': len(feedback_history),
            'average_quality': avg_quality,
            'final_quality': final_quality,
            'abstraction_trend': abstraction_trend,
            'informativeness_trend': informativeness_trend,
            'generalization_trend': generalization_trend,
            'iteration_metrics': iteration_metrics,
            'quality_improvement': final_quality - iteration_metrics[0].overall_quality if len(iteration_metrics) > 1 else 0.0
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        start_avg = sum(values[:len(values)//2]) / (len(values)//2)
        end_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = end_avg - start_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"

    def create_feedback_evaluation_report(self, metrics: FeedbackQualityMetrics) -> str:
        """Create a human-readable evaluation report."""
        def score_to_grade(score: float) -> str:
            if score >= 0.9: return "A"
            elif score >= 0.8: return "B" 
            elif score >= 0.7: return "C"
            elif score >= 0.6: return "D"
            else: return "F"
        
        report = f"""
Feedback Quality Evaluation Report
================================
Overall Quality: {metrics.overall_quality:.3f} (Grade: {score_to_grade(metrics.overall_quality)})

Component Scores:
  Abstraction:        {metrics.abstraction_score:.3f} (Grade: {score_to_grade(metrics.abstraction_score)})
  Informativeness:    {metrics.informativeness_score:.3f} (Grade: {score_to_grade(metrics.informativeness_score)})
  Generalization:     {metrics.generalization_score:.3f} (Grade: {score_to_grade(metrics.generalization_score)})
  Improvement:        {metrics.instruction_improvement_score:.3f} (Grade: {score_to_grade(metrics.instruction_improvement_score)})
  Overfitting Penalty: {metrics.overfitting_penalty:.3f}

Recommendations:
"""
        
        if metrics.abstraction_score < 0.7:
            report += "• Improve feedback abstraction to remove task-specific content\n"
        if metrics.informativeness_score < 0.7:
            report += "• Enhance feedback with more actionable structural information\n"
        if metrics.generalization_score < 0.7:
            report += "• Focus on generating more general-purpose instructions\n"
        if metrics.overfitting_penalty > 0.2:
            report += "• Reduce task-specific terms in evolved instructions\n"
        if metrics.overall_quality >= 0.8:
            report += "• Feedback quality is excellent - maintain current approach\n"
        
        return report


# Integration function for use in GEPA optimization
def evaluate_gepa_feedback_quality(optimizer_instance, feedback_samples: List[FeedbackResult]) -> Dict[str, Any]:
    """Evaluate GEPA's feedback quality during optimization.
    
    Args:
        optimizer_instance: The GEPA optimizer instance
        feedback_samples: Sample feedback results to evaluate
    
    Returns:
        Dictionary with quality metrics and recommendations
    """
    evaluator = FeedbackEvaluator()
    
    if not feedback_samples:
        return {'error': 'No feedback samples provided'}
    
    # Extract feedback evolution data
    feedback_history = []
    for i, feedback in enumerate(feedback_samples):
        # Simulate feedback evolution (in real use, this would track actual evolution)
        original_feedback = "Original detailed feedback with specific content"
        abstract_feedback = "Abstract structural feedback"
        evolved_instruction = f"Evolved instruction {i+1}"
        performance = sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0
        
        feedback_history.append((original_feedback, abstract_feedback, evolved_instruction, performance))
    
    return evaluator.evaluate_feedback_evolution(feedback_history)