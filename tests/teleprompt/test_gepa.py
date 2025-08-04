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
    Cohort,
    CandidatePool,
    Budget,
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

            result = optimizer.compile(student, simple_trainset)

            assert isinstance(result, Module)
            assert result is not student  # Should return a compiled copy
            assert hasattr(result, '_compiled')
            assert result._compiled is True


class TestComponentInterfaces:
    """Test that strategy interfaces work correctly."""

    def test_budget_interface(self):
        """Budget components should implement new interface."""
        from dspy.teleprompt.gepa.budget import LLMCallsBudget

        budget = LLMCallsBudget(100)

        # New interface methods
        assert hasattr(budget, 'spend_on_evaluation')
        assert hasattr(budget, 'spend_on_generation')
        assert hasattr(budget, 'spend_on_selection')
        assert hasattr(budget, 'get_remaining')

        # Magic comparison methods
        assert hasattr(budget, '__gt__')
        assert hasattr(budget, '__float__')

        # Test magic methods work
        assert budget > 0
        assert float(budget) == 100.0

    def test_generation_interface(self):
        """Generation components should implement required interface."""
        from dspy.teleprompt.gepa.generation import MutationGenerator

        strategy = MutationGenerator(mutation_rate=0.5)

        # Interface methods
        assert hasattr(strategy, 'generate')

    def test_selection_interface(self):
        """Selection components should implement required interface."""
        from dspy.teleprompt.gepa.selection import ParetoSelection

        strategy = ParetoSelection()

        # Interface methods
        assert hasattr(strategy, 'filter_candidates')
        assert hasattr(strategy, 'filter_scores')
        assert hasattr(strategy, 'filter_generation')
        assert hasattr(strategy, 'filter_generation_history')

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

            # Track the algorithm execution without mocking to avoid issues with reconfiguration
            result = optimizer.compile(student, simple_trainset)

            # Verify that optimization completed successfully
            assert result is not None
            assert hasattr(result, '_compiled')
            assert result._compiled == True

            # Verify that candidates were processed (shown by pool size in logs)
            assert optimizer.candidate_pool.size() > 0


class TestDataStructures:
    """Test core data structures work correctly."""

    def test_candidate_creation(self):
        """Candidates should be created with proper structure."""
        module = SimpleQA()
        candidate = Candidate(module=module, generation_number=1)

        assert candidate.module is module
        assert candidate.generation_number == 1
        assert not hasattr(candidate, 'candidate_id')  # No longer using IDs
        assert isinstance(candidate.task_scores, dict)

    def test_cohort_creation(self):
        """Cohorts should manage candidates correctly."""
        candidates = [Candidate(SimpleQA(), generation_number=0)]
        cohort = Cohort(candidates)

        assert cohort.size() == 1
        assert not cohort.is_empty()
        assert cohort.iteration_id == 0

    def test_candidate_pool_operations(self):
        """CandidatePool should manage candidates correctly."""
        pool = CandidatePool()
        candidate = Candidate(SimpleQA(), generation_number=0)

        # Add candidate
        pool.append(candidate)

        assert pool.size() == 1

        # Test that pool contains the candidate by checking size changes
        pool2 = CandidatePool()
        assert pool2.size() == 0
        pool2.append(candidate)
        assert pool2.size() == 1


class TestFactoryFunctions:
    """Test factory functions create valid GEPA instances."""

    def test_create_basic_gepa(self):
        """GEPA.create_basic should return working GEPA instance."""
        optimizer = GEPA.create_basic(simple_metric, max_calls=100)

        assert isinstance(optimizer, GEPA)
        assert hasattr(optimizer, 'budget')
        assert hasattr(optimizer, 'selector')
        assert hasattr(optimizer, 'generator')
        assert hasattr(optimizer, 'evaluator')



class TestGEPAIntegration:
    """Integration tests for complete GEPA workflows."""

    def test_basic_optimization_workflow(self, simple_trainset, dummy_lm):
        """Test basic optimization workflow end-to-end."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPA.create_basic(simple_metric, max_calls=30)

            # Should complete without errors
            result = optimizer.compile(student, simple_trainset)

            assert isinstance(result, Module)
            assert result._compiled is True
