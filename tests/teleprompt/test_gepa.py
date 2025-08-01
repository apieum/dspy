"""Tests for GEPA telepromter.

Following DSPy test patterns and BDD approach for GEPA implementation.
"""

import pytest
from unittest.mock import Mock, patch

import dspy
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.signatures.signature import make_signature
from dspy.teleprompt.gepa import (
    GEPA, 
    BudgetTracker, 
    ScoreMatrix,
    FeedbackResult,
    ParetoCandidateSelector,
    ReflectivePromptMutator,
    EnhancedFeedbackCollector,
    RoundRobinModuleSelector,
)
from dspy.utils.dummies import DummyLM


class SimpleQA(Module):
    """Simple QA program for testing."""
    
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")
        
    def forward(self, question):
        return self.answer(question=question)


def simple_metric(example, prediction, trace=None):
    """Simple metric for testing."""
    expected = example.answer.lower() if hasattr(example, 'answer') else ""
    actual = prediction.answer.lower() if hasattr(prediction, 'answer') else ""
    return 1.0 if expected == actual else 0.0


@pytest.fixture
def simple_trainset():
    """Simple training dataset for testing."""
    return [
        Example(question="What is 2+2?", answer="4"),
        Example(question="What color is the sky?", answer="blue"),
        Example(question="What is the capital of France?", answer="Paris"),
        Example(question="How many legs does a cat have?", answer="4"),
        Example(question="What planet do we live on?", answer="Earth"),
    ]


@pytest.fixture 
def dummy_lm():
    """DummyLM for predictable test responses."""
    return DummyLM([
        {"answer": "4"},
        {"answer": "blue"}, 
        {"answer": "Paris"},
        {"answer": "4"},
        {"answer": "Earth"},
        {"response": "Improved instruction: Answer questions accurately and concisely."}
    ])


class TestDataStructures:
    """Test core data structures."""
    
    def test_budget_tracker_initialization(self):
        """Budget tracker should initialize with zero usage."""
        budget = BudgetTracker(limit=100)
        assert budget.used == 0
        assert budget.limit == 100
        assert budget.has_budget() is True
        
    def test_budget_tracker_cost_tracking(self):
        """Budget tracker should correctly track different cost types."""
        budget = BudgetTracker(limit=100)
        
        budget.add_minibatch_cost(10)
        budget.add_reflection_cost(1)
        budget.add_pareto_cost(20)
        
        stats = budget.get_stats()
        assert stats['minibatch'] == 10
        assert stats['reflection'] == 1
        assert stats['pareto'] == 20
        assert stats['total_used'] == 31
        assert stats['remaining'] == 69
        
    def test_budget_tracker_limit_checking(self):
        """Budget tracker should correctly check budget limits."""
        budget = BudgetTracker(limit=10)
        assert budget.has_budget() is True
        
        budget.add_minibatch_cost(10)
        assert budget.has_budget() is False
        
    def test_score_matrix_operations(self):
        """Score matrix should handle candidate-task score management."""
        scores = ScoreMatrix()
        
        # Set scores
        scores.set_score(0, 0, 0.8)  # candidate 0, task 0, score 0.8
        scores.set_score(0, 1, 0.6)  # candidate 0, task 1, score 0.6
        scores.set_score(1, 0, 0.9)  # candidate 1, task 0, score 0.9
        
        # Get scores
        assert scores.get_score(0, 0) == 0.8
        assert scores.get_score(0, 1) == 0.6
        assert scores.get_score(1, 0) == 0.9
        assert scores.get_score(2, 0) is None  # Non-existent candidate
        
        # Compute averages
        assert scores.compute_average_score(0) == 0.7  # (0.8 + 0.6) / 2
        assert scores.compute_average_score(1) == 0.9  # 0.9 / 1
        assert scores.compute_average_score(2) == 0.0  # No scores
        
    def test_feedback_result_structure(self):
        """FeedbackResult should contain traces, diagnostics, and scores."""
        feedback = FeedbackResult(
            traces=[[], []],
            diagnostics=["Good", "Needs improvement"],
            scores=[0.8, 0.3]
        )
        
        assert len(feedback.traces) == 2
        assert len(feedback.diagnostics) == 2
        assert len(feedback.scores) == 2
        assert feedback.scores[0] == 0.8


class TestGEPABehavior:
    """Test GEPA behavioral requirements."""
    
    def test_gepa_initialization_with_defaults(self):
        """GEPA should initialize with default components."""
        gepa = GEPA(metric=simple_metric)
        
        assert gepa.metric == simple_metric
        assert gepa.minibatch_size == 3
        assert gepa.pareto_ratio == 0.67
        assert gepa.merge_enabled is False
        assert isinstance(gepa.candidate_selector, ParetoCandidateSelector)
        assert isinstance(gepa.prompt_mutator, ReflectivePromptMutator)
        assert isinstance(gepa.feedback_collector, EnhancedFeedbackCollector)
        assert isinstance(gepa.module_selector, RoundRobinModuleSelector)
        
    def test_gepa_initialization_with_custom_components(self):
        """GEPA should accept custom components via dependency injection.""" 
        custom_selector = Mock(spec=ParetoCandidateSelector)
        custom_mutator = Mock(spec=ReflectivePromptMutator)
        
        gepa = GEPA(
            metric=simple_metric,
            candidate_selector=custom_selector,
            prompt_mutator=custom_mutator
        )
        
        assert gepa.candidate_selector == custom_selector
        assert gepa.prompt_mutator == custom_mutator
        
    def test_gepa_rejects_compiled_student(self):
        """GEPA should reject pre-compiled student programs."""
        program = SimpleQA()
        program._compiled = True
        
        gepa = GEPA(metric=simple_metric)
        
        with pytest.raises(ValueError, match="should not be pre-compiled"):
            gepa.compile(program, trainset=[])
            
    def test_gepa_returns_compiled_program(self, simple_trainset, dummy_lm):
        """GEPA should return a compiled program."""
        with dspy.context(lm=dummy_lm):
            program = SimpleQA()
            gepa = GEPA(metric=simple_metric)
            
            # This will fail due to NotImplementedError, but we're testing structure
            with pytest.raises(NotImplementedError):
                compiled_program = gepa.compile(program, trainset=simple_trainset, budget=50)
                
    def test_gepa_preserves_original_program(self, simple_trainset, dummy_lm):
        """GEPA should not modify the original student program."""
        with dspy.context(lm=dummy_lm):
            original_program = SimpleQA()
            original_compiled_state = getattr(original_program, '_compiled', False)
            
            gepa = GEPA(metric=simple_metric)
            
            # This will fail due to NotImplementedError, but we're testing structure
            with pytest.raises(NotImplementedError):
                gepa.compile(original_program, trainset=simple_trainset, budget=50)
                
            # Original program should be unchanged
            assert getattr(original_program, '_compiled', False) == original_compiled_state


class TestComponentInterfaces:
    """Test that component interfaces are properly defined."""
    
    def test_candidate_selector_interface(self):
        """CandidateSelector should define select_candidate method."""
        selector = ParetoCandidateSelector()
        
        with pytest.raises(NotImplementedError):
            selector.select_candidate([], ScoreMatrix())
            
    def test_prompt_mutator_interface(self):
        """PromptMutator should define mutate_signature method."""
        mutator = ReflectivePromptMutator()
        signature = make_signature("question -> answer")
        feedback = FeedbackResult(traces=[], diagnostics=[], scores=[])
        
        with pytest.raises(NotImplementedError):
            mutator.mutate_signature(signature, feedback)
            
    def test_feedback_collector_interface(self):
        """FeedbackCollector should define collect_feedback method."""
        collector = EnhancedFeedbackCollector()
        program = SimpleQA()
        examples = []
        
        with pytest.raises(NotImplementedError):
            collector.collect_feedback(program, examples, simple_metric)
            
    def test_module_selector_interface(self):
        """ModuleSelector should define select_module method."""
        selector = RoundRobinModuleSelector()
        program = SimpleQA()
        
        with pytest.raises(NotImplementedError):
            selector.select_module(program)


class TestGEPAAlgorithmStructure:
    """Test that GEPA implements the correct algorithm structure."""
    
    def test_gepa_dataset_splitting_structure(self, simple_trainset):
        """GEPA should split dataset into feedback and Pareto sets."""
        gepa = GEPA(metric=simple_metric)
        
        with pytest.raises(NotImplementedError, match="Dataset splitting"):
            gepa._split_dataset(simple_trainset)
            
    def test_gepa_pareto_evaluation_structure(self):
        """GEPA should evaluate candidates on Pareto set."""
        gepa = GEPA(metric=simple_metric)
        candidates = [SimpleQA()]
        pareto_data = []
        scores = ScoreMatrix() 
        budget = BudgetTracker(limit=100)
        
        with pytest.raises(NotImplementedError, match="Pareto evaluation"):
            gepa._evaluate_candidates_on_pareto(candidates, pareto_data, scores, budget)
            
    def test_gepa_candidate_promotion_structure(self):
        """GEPA should have candidate promotion logic."""
        gepa = GEPA(metric=simple_metric)
        new_candidate = SimpleQA()
        parent_candidate = SimpleQA()
        feedback_data = []
        budget = BudgetTracker(limit=100)
        
        with pytest.raises(NotImplementedError, match="promotion logic"):
            gepa._should_promote_candidate(new_candidate, parent_candidate, feedback_data, budget)
            
    def test_gepa_best_candidate_selection_structure(self):
        """GEPA should select best candidate from scores."""
        gepa = GEPA(metric=simple_metric)
        candidates = [SimpleQA()]
        scores = ScoreMatrix()
        
        with pytest.raises(NotImplementedError, match="Best candidate selection"):
            gepa._select_best_candidate(candidates, scores)


if __name__ == "__main__":
    pytest.main([__file__])