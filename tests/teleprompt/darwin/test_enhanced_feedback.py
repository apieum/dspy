"""Test Enhanced Feedback Function (μf) implementation."""

import pytest
from unittest.mock import Mock

import dspy
from dspy.teleprompt.darwin.generation.feedback import FeedbackProvider
from dspy.teleprompt.darwin.generation.enhanced_metrics import (
    code_evaluation_metric,
    math_problem_metric,
    text_classification_metric,
    qa_accuracy_metric
)


class TestEnhancedFeedbackFunction:
    """Test μf-compliant enhanced feedback functionality."""


    def test_enhanced_metric_with_rich_feedback(self):
        """Test μf-compliant metrics with rich diagnostic feedback."""
        def enhanced_metric(example, prediction, trace=None):
            # Return (score, rich_feedback) tuple
            if example.answer == prediction.answer:
                return (1.0, "Perfect match with expected answer")
            else:
                return (0.0, "Answer mismatch: expected 'correct' but got 'wrong'")

        provider = FeedbackProvider(metric=enhanced_metric)

        # Test successful case
        example = dspy.Example(question="test", answer="correct")
        prediction = Mock()
        prediction.answer = "correct"

        score, diagnostic = provider.evaluate(example, prediction)

        assert score == 1.0
        assert "Score: 1.00 (SUCCESS)" in diagnostic
        assert "Evaluator Feedback: Perfect match with expected answer" in diagnostic

        # Test failure case
        prediction.answer = "wrong"
        score, diagnostic = provider.evaluate(example, prediction)

        assert score == 0.0
        assert "Score: 0.00 (FAILURE)" in diagnostic
        assert "Evaluator Feedback: Answer mismatch: expected 'correct' but got 'wrong'" in diagnostic

    def test_code_evaluation_metric_syntax_error(self):
        """Test code evaluation metric with syntax errors."""
        example = dspy.Example(question="Write a function", answer="def add(a,b): return a+b")
        prediction = Mock()
        prediction.code = "def add(a, b return a + b"  # Missing colon

        score, feedback = code_evaluation_metric(example, prediction)

        assert score == 0.0
        assert "Syntax error" in feedback
        assert "missing" in feedback.lower() or "colon" in feedback.lower()

    def test_code_evaluation_metric_success(self):
        """Test code evaluation metric with correct code."""
        example = dspy.Example(question="Write a function", answer="def add(a,b): return a+b")
        prediction = Mock()
        prediction.code = "def add(a, b):\n    return a + b"

        score, feedback = code_evaluation_metric(example, prediction)

        assert score >= 0.9  # High score for good code
        assert "compiles successfully" in feedback.lower()

    def test_math_problem_metric_correct(self):
        """Test math problem metric with correct answer."""
        example = dspy.Example(question="What is 2+3?", answer="5")
        prediction = Mock()
        prediction.answer = "The answer is 5"

        score, feedback = math_problem_metric(example, prediction)

        assert score == 1.0
        assert "Correct answer: 5" in feedback

    def test_math_problem_metric_close(self):
        """Test math problem metric with close but inexact answer."""
        example = dspy.Example(question="What is sqrt(2)?", answer="1.414")
        prediction = Mock()
        prediction.answer = "1.41"  # Close but not exact

        score, feedback = math_problem_metric(example, prediction)

        assert 0.7 <= score <= 0.9  # Partial credit
        assert "close" in feedback.lower()
        assert "not exact" in feedback.lower()

    def test_text_classification_metric_correct(self):
        """Test text classification metric with correct classification."""
        example = dspy.Example(text="I love this!", label="positive")
        prediction = Mock()
        prediction.classification = "positive"

        score, feedback = text_classification_metric(example, prediction)

        assert score == 1.0
        assert "Correct classification: 'positive'" in feedback

    def test_text_classification_metric_opposite(self):
        """Test text classification metric with opposite classification."""
        example = dspy.Example(text="I love this!", label="positive")
        prediction = Mock()
        prediction.classification = "negative"

        score, feedback = text_classification_metric(example, prediction)

        assert score == 0.0
        assert "opposite" in feedback.lower()
        assert "review the input text more carefully" in feedback.lower()

    def test_qa_accuracy_metric_exact_match(self):
        """Test QA accuracy metric with exact match."""
        example = dspy.Example(question="What is the capital of France?", answer="Paris")
        prediction = Mock()
        prediction.answer = "Paris"

        score, feedback = qa_accuracy_metric(example, prediction)

        assert score == 1.0
        assert "Exact answer match" in feedback

    def test_qa_accuracy_metric_verbose(self):
        """Test QA accuracy metric with verbose but correct answer."""
        example = dspy.Example(question="What is the capital of France?", answer="Paris")
        prediction = Mock()
        prediction.answer = "The capital of France is Paris, which is a beautiful city."

        score, feedback = qa_accuracy_metric(example, prediction)

        assert score >= 0.8  # High score but not perfect
        assert "overly verbose" in feedback.lower()

    def test_combined_feedback_provider_with_additional_function(self):
        """Test FeedbackProvider with both μf metric and additional feedback function."""
        def enhanced_metric(example, prediction, trace=None):
            return (0.6, "Metric says: partially correct")

        def additional_feedback_func(example, prediction, trace, module_idx):
            return "Additional feedback: consider improving X and Y"

        provider = FeedbackProvider(
            metric=enhanced_metric,
            feedback_function=additional_feedback_func
        )

        example = dspy.Example(question="test", answer="answer")
        prediction = Mock()
        prediction.answer = "answer"

        score, diagnostic = provider.evaluate(example, prediction)

        assert score == 0.6
        # Should contain both the metric feedback and additional feedback
        assert "Evaluator Feedback: Metric says: partially correct" in diagnostic
        assert "Additional feedback: consider improving X and Y" in diagnostic

    def test_enhanced_feedback_in_reflection_context(self):
        """Test how enhanced feedback appears in reflection formatting."""
        from dspy.teleprompt.darwin.generation.prompt_mutator import ReflectivePromptMutator
        from dspy.teleprompt.darwin.generation.reflection_strategy import GEPAReflection
        from dspy.teleprompt.darwin.evaluation.feedback import FeedbackResult

        # Create enhanced metric
        def code_metric(example, prediction, trace=None):
            return (0.0, "Compilation failed with SyntaxError: missing colon on line 1")

        provider = FeedbackProvider(metric=code_metric)
        mutator = ReflectivePromptMutator(GEPAReflection())

        # Simulate feedback result with enhanced diagnostics
        feedback = FeedbackResult(
            scores=[0.0],
            diagnostics=["Score: 0.0 (FAILURE) | Evaluator Feedback: Compilation failed with SyntaxError: missing colon on line 1"],
            traces=[[(Mock(), {"code": "def add(a, b return a + b"}, {"result": "error"})]]
        )

        # Format feedback for reflection
        formatted = mutator._format_feedback_for_reflection(feedback, 0)

        # Should contain rich diagnostic information
        assert "SyntaxError" in formatted
        assert "missing colon" in formatted
        assert "line 1" in formatted

        # This rich feedback enables much more targeted reflection
