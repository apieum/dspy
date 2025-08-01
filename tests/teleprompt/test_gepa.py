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
    EvaluationTrace,
    ModuleFeedback,
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
        Example(question="What is 2+2?", answer="4").with_inputs("question"),
        Example(question="What color is the sky?", answer="blue").with_inputs("question"),
        Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        Example(question="How many legs does a cat have?", answer="4").with_inputs("question"),
        Example(question="What planet do we live on?", answer="Earth").with_inputs("question"),
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
        """FeedbackResult should contain enhanced feedback with μf function support."""
        feedback = FeedbackResult(
            traces=[[], []],
            diagnostics=["Good", "Needs improvement"],
            scores=[0.8, 0.3],
            evaluation_traces=[],  # Enhanced: Rich evaluation traces
            module_feedback=[],    # Enhanced: Module-level feedback
            feedback_text=[]       # Enhanced: Textual feedback
        )
        
        assert len(feedback.traces) == 2
        assert len(feedback.diagnostics) == 2
        assert len(feedback.scores) == 2
        assert feedback.scores[0] == 0.8
        
        # Test enhanced fields
        assert hasattr(feedback, 'evaluation_traces')
        assert hasattr(feedback, 'module_feedback')
        assert hasattr(feedback, 'feedback_text')


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
            
            # Now GEPA should actually work - test that it returns a compiled program
            compiled_program = gepa.compile(program, trainset=simple_trainset, budget=50)
            
            assert compiled_program is not None
            assert hasattr(compiled_program, '_compiled')
            assert compiled_program._compiled is True
                
    def test_gepa_preserves_original_program(self, simple_trainset, dummy_lm):
        """GEPA should not modify the original student program."""
        with dspy.context(lm=dummy_lm):
            original_program = SimpleQA()
            original_compiled_state = getattr(original_program, '_compiled', False)
            
            gepa = GEPA(metric=simple_metric)
            
            # Now GEPA should actually work
            compiled_program = gepa.compile(original_program, trainset=simple_trainset, budget=50)
                
            # Original program should be unchanged
            assert getattr(original_program, '_compiled', False) == original_compiled_state
            # Compiled program should be different instance
            assert original_program is not compiled_program


class TestComponentInterfaces:
    """Test that component interfaces are properly defined."""
    
    def test_candidate_selector_interface(self):
        """CandidateSelector should work with candidates and scores."""
        selector = ParetoCandidateSelector()
        program = SimpleQA()
        scores = ScoreMatrix()
        
        # Should work with empty candidates
        result = selector.select_candidate([program], scores)
        assert isinstance(result, int)
        assert 0 <= result < 1
            
    def test_prompt_mutator_interface(self):
        """PromptMutator should mutate signatures based on feedback."""
        mutator = ReflectivePromptMutator()
        signature = make_signature("question -> answer")
        feedback = FeedbackResult(traces=[], diagnostics=[], scores=[], evaluation_traces=[], module_feedback=[], feedback_text=[])
        
        # Should return a signature (might be same or different)
        result = mutator.mutate_signature(signature, feedback)
        assert result is not None
        assert hasattr(result, 'fields')
            
    def test_feedback_collector_interface(self):
        """FeedbackCollector should collect feedback from program execution."""
        collector = EnhancedFeedbackCollector()
        program = SimpleQA()
        examples = []
        
        # Should work with empty examples
        result = collector.collect_feedback(program, examples, simple_metric)
        assert isinstance(result, FeedbackResult)
        assert result.traces == []
        assert result.diagnostics == []
        assert result.scores == []
        
        # Test enhanced fields
        assert result.evaluation_traces == []
        assert result.module_feedback == []
        assert result.feedback_text == []


class TestEnhancedFeedbackFunction:
    """Test enhanced feedback function μf implementation."""
    
    def test_evaluation_trace_structure(self):
        """EvaluationTrace should contain rich evaluation information."""
        trace = EvaluationTrace(
            execution_steps=["Step 1: Initialize", "Step 2: Process"],
            compilation_errors=[],
            intermediate_outputs=[{"step1": "result"}],
            module_outputs={0: "output1", 1: "output2"},
            reasoning_chains=["First, analyze...", "Then, conclude..."],
            tool_calls=[{"tool": "search", "query": "test"}],
            error_messages=[],
            performance_metrics={"execution_time": 0.5, "score": 0.8}
        )
        
        assert len(trace.execution_steps) == 2
        assert len(trace.intermediate_outputs) == 1
        assert len(trace.module_outputs) == 2
        assert len(trace.reasoning_chains) == 2
        assert len(trace.tool_calls) == 1
        assert trace.performance_metrics["score"] == 0.8
    
    def test_module_feedback_structure(self):
        """ModuleFeedback should contain per-module performance information."""
        feedback = ModuleFeedback(
            module_id=0,
            module_name="QueryModule",
            input_data="What is the capital?",
            output_data="Paris",
            execution_time=0.2,
            success=True,
            error_message=None,
            intermediate_reasoning=["Analyzing query...", "Retrieving answer..."],
            confidence_score=0.9
        )
        
        assert feedback.module_id == 0
        assert feedback.module_name == "QueryModule"
        assert feedback.success is True
        assert feedback.confidence_score == 0.9
        assert len(feedback.intermediate_reasoning) == 2
    
    def test_enhanced_feedback_collector_with_examples(self, simple_trainset, dummy_lm):
        """Enhanced feedback collector should provide rich evaluation traces."""
        with dspy.context(lm=dummy_lm):
            collector = EnhancedFeedbackCollector(
                collect_module_feedback=True,
                collect_evaluation_traces=True
            )
            program = SimpleQA()
            examples = simple_trainset[:2]  # Use first 2 examples
            
            result = collector.collect_feedback(program, examples, simple_metric)
            
            # Test basic fields
            assert isinstance(result, FeedbackResult)
            assert len(result.traces) == 2
            assert len(result.diagnostics) == 2
            assert len(result.scores) == 2
            
            # Test enhanced fields
            assert len(result.evaluation_traces) == 2
            assert len(result.module_feedback) == 2
            assert len(result.feedback_text) == 2
            
            # Test evaluation trace content
            for eval_trace in result.evaluation_traces:
                assert isinstance(eval_trace, EvaluationTrace)
                assert hasattr(eval_trace, 'execution_steps')
                assert hasattr(eval_trace, 'performance_metrics')
            
            # Test feedback text content
            for feedback_text in result.feedback_text:
                assert isinstance(feedback_text, str)
                assert len(feedback_text) > 0
    
    def test_domain_specific_feedback(self):
        """Test domain-specific feedback handling."""
        collector = EnhancedFeedbackCollector()
        
        # Register a custom domain handler
        def code_feedback_handler(example, prediction, trace):
            return {
                'compilation_status': 'success',
                'code_quality': 'high',
                'performance': 'optimized'
            }
        
        collector.register_domain_handler('code', code_feedback_handler)
        assert 'code' in collector.domain_handlers
        
        # Test domain detection
        code_example = Example(
            question="Write a Python function to sort a list",
            code="def sort_list(lst): return sorted(lst)"
        ).with_inputs("question")
        
        domain = collector._detect_domain(code_example)
        assert domain == 'code'
            
    def test_module_selector_interface(self):
        """ModuleSelector should select modules from programs."""
        selector = RoundRobinModuleSelector()
        program = SimpleQA()
        
        # Should return valid module index
        result = selector.select_module(program)
        assert isinstance(result, int)
        assert 0 <= result < len(program.predictors())


class TestGEPAAlgorithmStructure:
    """Test that GEPA implements the correct algorithm structure."""
    
    def test_gepa_dataset_splitting_structure(self, simple_trainset):
        """GEPA should split dataset into feedback and Pareto sets."""
        gepa = GEPA(metric=simple_metric)
        
        # Should split dataset according to pareto_ratio
        feedback_data, pareto_data = gepa._split_dataset(simple_trainset)
        
        assert len(feedback_data) + len(pareto_data) == len(simple_trainset)
        assert len(pareto_data) == int(len(simple_trainset) * gepa.pareto_ratio)
        assert len(feedback_data) == len(simple_trainset) - len(pareto_data)
            
    def test_gepa_pareto_evaluation_structure(self):
        """GEPA should evaluate candidates on Pareto set."""
        gepa = GEPA(metric=simple_metric)
        candidates = [SimpleQA()]
        pareto_data = []
        scores = ScoreMatrix() 
        budget = BudgetTracker(limit=100)
        
        # Should handle empty pareto_data gracefully
        gepa._evaluate_candidates_on_pareto(candidates, pareto_data, scores, budget)
        
        # Should not crash with valid inputs
        assert True  # If we get here, the method worked
            
    def test_gepa_candidate_promotion_structure(self):
        """GEPA should have candidate promotion logic."""
        gepa = GEPA(metric=simple_metric)
        new_candidate = SimpleQA()
        parent_candidate = SimpleQA()
        feedback_data = []
        budget = BudgetTracker(limit=100)
        
        # Should return boolean decision
        result = gepa._should_promote_candidate(new_candidate, parent_candidate, feedback_data, budget)
        assert isinstance(result, bool)
        
        # With empty feedback_data, should return False
        assert result is False
            
    def test_gepa_best_candidate_selection_structure(self):
        """GEPA should select best candidate from scores."""
        gepa = GEPA(metric=simple_metric)
        candidates = [SimpleQA()]
        scores = ScoreMatrix()
        
        # Should return a candidate from the list
        result = gepa._select_best_candidate(candidates, scores)
        assert result in candidates
        
        # With single candidate, should return that candidate
        assert result is candidates[0]


if __name__ == "__main__":
    pytest.main([__file__])