"""Tests for GEPA telepromter.

Following DSPy test patterns and BDD approach for GEPA implementation.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Iterable, List

import dspy
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.signatures.signature import make_signature
from dspy.teleprompt.gepa import (
    GEPA, 
    Candidate,
    Generation,
    ScoreMatrix,
    CandidatePool,
    Budget,
    Scoring,
    Selection,
    Generator,
    Evaluator,
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


class TestGEPABehavior:
    """Test GEPA core behavior and algorithm structure."""
    
    def test_gepa_returns_compiled_program(self, simple_trainset, dummy_lm):
        """GEPA should return a compiled program when given valid inputs."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPA.create_basic(simple_metric, max_calls=50)
            
            result = optimizer.compile(student, simple_trainset[:3], simple_trainset[3:])
            
            assert isinstance(result, Module)
            assert result is not student  # Should return a compiled copy
            assert hasattr(result, '_compiled')
            assert result._compiled is True


class TestComponentInterfaces:
    """Test that strategy interfaces work correctly."""
    
    def test_budget_interface(self):
        """Budget components should implement required interface."""
        from dspy.teleprompt.gepa.budget import LLMCallsBudget
        
        budget = LLMCallsBudget(100)
        
        # Interface methods
        assert hasattr(budget, 'can_afford')
        assert hasattr(budget, 'consume')
        assert hasattr(budget, 'is_empty')
        
        # Behavior
        assert budget.can_afford(10, "test")
        budget.consume(50, "test")
        assert budget.can_afford(10, "test")
        budget.consume(50, "test")
        assert budget.is_empty()
    
    def test_scoring_interface(self):
        """Scoring components should implement required interface."""
        from dspy.teleprompt.gepa.scoring import ParetoScoring
        
        strategy = ParetoScoring(simple_metric)
        
        # Interface methods
        assert hasattr(strategy, 'calculate_scores')
        
    def test_generation_interface(self):
        """Generation components should implement required interface."""
        from dspy.teleprompt.gepa.generation import MutationGenerator
        
        strategy = MutationGenerator(mutation_rate=0.5)
        
        # Interface methods
        assert hasattr(strategy, 'generate')
        
    def test_filtering_interface(self):
        """Filtering components should implement required interface."""
        from dspy.teleprompt.gepa.selection import ElitistSelection
        
        strategy = ElitistSelection(keep_top_n=5)
        
        # Interface methods
        assert hasattr(strategy, 'filter')
        
    def test_evaluation_interface(self):
        """Evaluation components should implement required interface."""
        from dspy.teleprompt.gepa.evaluation import PromotionEvaluator
        
        strategy = PromotionEvaluator(simple_metric, promotion_threshold=0.6)
        
        # Interface methods
        assert hasattr(strategy, 'evaluate')


class TestGEPAAlgorithmStructure:
    """Test the GEPA algorithm follows the correct structure."""
    
    def test_gepa_algorithm_phases(self, simple_trainset, dummy_lm):
        """GEPA should follow the defined algorithm phases."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPA.create_basic(simple_metric, max_calls=20)
            
            # Mock the optimization components to track calls
            with patch.object(optimizer.scoring, 'calculate_scores', wraps=optimizer.scoring.calculate_scores) as mock_scoring, \
                 patch.object(optimizer.selection, 'filter', wraps=optimizer.selection.filter) as mock_filtering, \
                 patch.object(optimizer.generator, 'generate', wraps=optimizer.generator.generate) as mock_generation, \
                 patch.object(optimizer.evaluator, 'evaluate', wraps=optimizer.evaluator.evaluate) as mock_evaluation:
                
                result = optimizer.compile(student, simple_trainset[:2], simple_trainset[2:])
                
                # Verify algorithm phases were called
                assert mock_scoring.called
                assert mock_filtering.called  
                assert mock_generation.called
                assert mock_evaluation.called


class TestDataStructures:
    """Test core data structures work correctly."""
    
    def test_candidate_creation(self):
        """Candidates should be created with proper structure."""
        module = SimpleQA()
        candidate = Candidate(module=module, generation_number=1)
        
        assert candidate.module is module
        assert candidate.generation_number == 1
        assert candidate.candidate_id is None  # Assigned by pool
        assert isinstance(candidate.task_scores, list)
        
    def test_generation_creation(self):
        """Generations should manage candidates correctly."""
        candidates = [Candidate(SimpleQA(), generation_number=0)]
        generation = Generation(candidates, generation_id=0)
        
        assert generation.size() == 1
        assert not generation.is_empty()
        
    def test_candidate_pool_operations(self):
        """CandidatePool should manage candidates correctly."""
        pool = CandidatePool()
        candidate = Candidate(SimpleQA(), generation_number=0)
        
        # Add candidate
        pool.add_candidate(candidate)
        
        assert pool.size() == 1
        assert candidate.candidate_id is not None
        
        # Retrieve candidate
        retrieved = pool.get_candidate(candidate.candidate_id)
        assert retrieved is candidate


class TestFactoryFunctions:
    """Test factory functions create valid GEPA instances."""
    
    def test_create_basic_gepa(self):
        """GEPA.create_basic should return working GEPA instance."""
        optimizer = GEPA.create_basic(simple_metric, max_calls=100)
        
        assert isinstance(optimizer, GEPA)
        assert hasattr(optimizer, 'budget')
        assert hasattr(optimizer, 'scoring')
        assert hasattr(optimizer, 'selection')
        assert hasattr(optimizer, 'generator')
        assert hasattr(optimizer, 'evaluator')
        
    def test_create_diversity_gepa(self):
        """GEPA.create_diversity should return working GEPA instance."""
        optimizer = GEPA.create_diversity(simple_metric, max_calls=100)
        
        assert isinstance(optimizer, GEPA)
        # Should use diversity selection
        from dspy.teleprompt.gepa.selection import DiversitySelection
        assert isinstance(optimizer.selection, DiversitySelection)


class TestGEPAIntegration:
    """Integration tests for complete GEPA workflows."""
    
    def test_basic_optimization_workflow(self, simple_trainset, dummy_lm):
        """Test basic optimization workflow end-to-end."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPA.create_basic(simple_metric, max_calls=30)
            
            # Should complete without errors
            result = optimizer.compile(student, simple_trainset[:3], simple_trainset[3:])
            
            assert isinstance(result, Module)
            assert result._compiled is True
            
    def test_diversity_optimization_workflow(self, simple_trainset, dummy_lm):
        """Test diversity-focused optimization workflow."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPA.create_diversity(simple_metric, max_calls=30)
            
            # Should complete without errors
            result = optimizer.compile(student, simple_trainset[:3], simple_trainset[3:])
            
            assert isinstance(result, Module)
            assert result._compiled is True