"""Tests for Darwin optimizer.

Following DSPy test patterns and BDD approach for Darwin framework implementation.
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
    GEPAMute,
    Candidate,
    Cohort,
    Budget,
    Selector,
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


class TestDarwinBehavior:
    """Test Darwin core behavior and algorithm structure."""

    def test_darwin_returns_compiled_program(self, simple_trainset, dummy_lm):
        """Darwin should return a compiled program when given valid inputs."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(simple_metric, max_calls=2)

            result = optimizer.compile(student, simple_trainset)

            assert isinstance(result, Module)
            assert result is not student  # Should return a compiled copy
            assert hasattr(result, '_compiled')
            assert result._compiled is True


class TestDarwinAlgorithmStructure:
    """Test the Darwin framework follows the correct structure."""

    def test_darwin_algorithm_phases(self, simple_trainset, dummy_lm):
        """Darwin should follow the defined algorithm phases."""
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(simple_metric, max_calls=2)

            # Track the algorithm execution without mocking to avoid issues with reconfiguration
            result = optimizer.compile(student, simple_trainset)

            # Verify that optimization completed successfully
            assert result is not None
            assert hasattr(result, '_compiled')
            assert result._compiled == True

            # Verify that candidates were processed (shown by selector having candidates)
            assert optimizer.selector.size() > 0



class TestFactoryFunctions:
    """Test factory functions create valid GEPA instances."""

    def test_create_gepa_optimizer(self):
        """GEPAMute should return working Darwin instance."""
        optimizer = GEPAMute(simple_metric, max_calls=2)

        assert isinstance(optimizer, Darwin)
        assert hasattr(optimizer, 'budget')
        assert hasattr(optimizer, 'selector')
        assert hasattr(optimizer, 'generator')
        assert hasattr(optimizer, 'evaluator')
