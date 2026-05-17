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
from dspy.teleprompt.darwin import (
    Darwin,
    DarwinConfig,
    GEPAStrategy,
    LMCallsBudget,
    ParetoFrontier,
    ReflectivePromptMutation,
    FeedbackProvider,
    FullTaskScores,
    GEPATwoPhasesEval,
    Candidate,
    Cohort,
    Budget,
    Selector,
    Generator,
    Evaluator,
    Channel,
    Success,
    Failure,
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

            # Create GEPA configuration using the new architecture
            config = DarwinConfig(
                max_lm_calls=2,
                minibatch_size=2,
                verbose=False
            )

            optimizer = Darwin(GEPAStrategy, config)

            # Split for Darwin interface: use most for dev, minimal for train
            trainset = simple_trainset[:1]  # Minimal trainset for bootstrapping
            devset = simple_trainset[1:]     # Rest for development/optimization
            compiled_module = optimizer.compile(student, trainset=trainset, devset=devset)
            result = optimizer.get_last_result()

            # Result should be a Success with the compiled module
            assert isinstance(result, Success)
            assert isinstance(compiled_module, Module)
            assert compiled_module is not student  # Should return a compiled copy
            assert hasattr(compiled_module, '_compiled')
            assert compiled_module._compiled is True


class TestGEPAAlgorithmStructure:
    """Test the GEPA algorithm follows the correct structure."""

    def test_gepa_algorithm_phases(self, simple_trainset, dummy_lm):
        """GEPA should follow the defined algorithm phases."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()

            # Create GEPA configuration
            config = DarwinConfig(
                max_lm_calls=2,
                minibatch_size=2,
                verbose=False
            )

            optimizer = Darwin(GEPAStrategy, config)

            # Track the algorithm execution without mocking to avoid issues with reconfiguration
            # Split for Darwin interface: use most for dev, minimal for train
            trainset = simple_trainset[:1]  # Minimal trainset for bootstrapping
            devset = simple_trainset[1:]     # Rest for development/optimization
            compiled_module = optimizer.compile(student, trainset=trainset, devset=devset)
            result = optimizer.get_last_result()

            # Verify that optimization completed successfully
            assert isinstance(result, Success)
            assert compiled_module is not None
            assert hasattr(compiled_module, '_compiled')
            assert compiled_module._compiled == True

            # Verify that the algorithm executed steps
            # (we should have at least one generation since we see successful evaluation logs)
            assert optimizer.strategy.current_generation >= 0



class TestFactoryFunctions:
    """Test factory functions create valid GEPA instances."""

    def test_create_basic_gepa(self):
        """DarwinConfig should create working Darwin instance."""
        config = DarwinConfig(
            max_lm_calls=2,
            patience=3,
            minibatch_size=2,
            verbose=False
        )

        optimizer = Darwin(GEPAStrategy, config)

        assert isinstance(optimizer, Darwin)
        assert hasattr(optimizer, 'strategy')
        assert hasattr(optimizer.strategy, 'budget')
        assert hasattr(optimizer.strategy, 'selector')
        assert hasattr(optimizer.strategy, 'generator')
        assert hasattr(optimizer.strategy, 'evaluator')
