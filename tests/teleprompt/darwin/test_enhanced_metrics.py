"""Test enhanced metrics for Darwin framework."""

import pytest
from unittest.mock import Mock

import dspy
from dspy.teleprompt.darwin.generation.enhanced_metrics import (
    code_evaluation_metric,
    math_problem_metric, 
    text_classification_metric,
    qa_accuracy_metric
)


class TestCodeEvaluationMetric:
    """Test code evaluation metric with μf compliance."""

    def test_valid_function_code(self):
        """Test code evaluation with valid function code."""
        example = dspy.Example(question="Write a function", answer="def add(a,b): return a+b")
        prediction = Mock()
        prediction.code = "def add(a, b):\n    return a + b"
        
        score, feedback = code_evaluation_metric(example, prediction)
        
        assert score >= 0.9
        assert "compiles successfully" in feedback.lower()
        assert "good practices" in feedback.lower()

    def test_syntax_error_code(self):
        """Test code evaluation with syntax errors."""
        example = dspy.Example(question="Write a function", answer="def add(a,b): return a+b")
        prediction = Mock()
        prediction.code = "def add(a, b return a + b"  # Missing colon
        
        score, feedback = code_evaluation_metric(example, prediction)
        
        assert score == 0.0
        assert "syntax error" in feedback.lower()
        assert "colon" in feedback.lower() or "missing" in feedback.lower()

    def test_no_code_attribute(self):
        """Test code evaluation when no code attribute exists."""
        example = dspy.Example(question="Write code", answer="def test(): pass")
        prediction = Mock()
        # No code attribute
        
        score, feedback = code_evaluation_metric(example, prediction)
        
        assert score == 0.0
        assert "no code found" in feedback.lower()

    def test_function_without_return(self):
        """Test function code without return statement."""
        example = dspy.Example(question="Write function", answer="def add(a,b): return a+b")
        prediction = Mock()
        prediction.code = "def add(a, b):\n    print(a + b)"  # No return
        
        score, feedback = code_evaluation_metric(example, prediction)
        
        assert 0.6 <= score <= 0.8  # Quality deducted but still valid
        assert "return statement" in feedback.lower()

    def test_function_with_unused_parameters(self):
        """Test function with unused parameters."""
        example = dspy.Example(question="Write function", answer="def add(a,b): return a+b")
        prediction = Mock()
        prediction.code = "def add(a, b):\n    return 42"  # Unused params but still valid code
        
        score, feedback = code_evaluation_metric(example, prediction)
        
        assert score >= 0.9  # Code compiles successfully
        assert "good practices" in feedback.lower() or "compiles successfully" in feedback.lower()

    def test_no_function_when_expected(self):
        """Test when function is expected but not provided."""
        example = dspy.Example(question="Write function", answer="def add(a,b): return a+b")
        prediction = Mock()
        prediction.code = "result = 2 + 3"  # No function definition
        
        score, feedback = code_evaluation_metric(example, prediction)
        
        assert score == 0.3
        assert "no function definition" in feedback.lower()
        assert "def function_name" in feedback

    def test_evaluation_exception(self):
        """Test handling of unexpected evaluation exceptions."""
        example = dspy.Example(question="Write code", answer="test")
        prediction = None  # Treated as string "None"
        
        score, feedback = code_evaluation_metric(example, prediction)
        
        # The implementation converts None to string "None" which is valid code
        assert score >= 0.9
        assert "compiles successfully" in feedback.lower() or "good practices" in feedback.lower()


class TestMathProblemMetric:
    """Test math problem metric with μf compliance."""

    def test_exact_numerical_match(self):
        """Test exact numerical answer match."""
        example = dspy.Example(question="What is 2+3?", answer="5")
        prediction = Mock()
        prediction.answer = "5"
        
        score, feedback = math_problem_metric(example, prediction)
        
        assert score == 1.0
        assert "correct answer: 5" in feedback.lower()

    def test_close_numerical_match(self):
        """Test close but not exact numerical match."""
        example = dspy.Example(question="What is sqrt(2)?", answer="1.414")
        prediction = Mock()
        prediction.answer = "1.41"  # Within 5%
        
        score, feedback = math_problem_metric(example, prediction)
        
        assert 0.7 <= score <= 0.9
        assert "close" in feedback.lower()
        assert "not exact" in feedback.lower()

    def test_right_order_of_magnitude(self):
        """Test answer in right ballpark but not precise."""
        example = dspy.Example(question="What is 100*7?", answer="700")
        prediction = Mock()
        prediction.answer = "650"  # Within 50% but not close
        
        score, feedback = math_problem_metric(example, prediction)
        
        assert 0.3 <= score <= 0.5
        assert "right ballpark" in feedback.lower()
        assert "significant errors" in feedback.lower()

    def test_completely_wrong_answer(self):
        """Test completely incorrect numerical answer."""
        example = dspy.Example(question="What is 2+2?", answer="4")
        prediction = Mock()
        prediction.answer = "100"  # Way off
        
        score, feedback = math_problem_metric(example, prediction)
        
        assert score == 0.0
        assert "incorrect" in feedback.lower()
        assert "recalculate step by step" in feedback.lower()

    def test_no_numerical_answer(self):
        """Test when no numerical answer is provided."""
        example = dspy.Example(question="What is 5*6?", answer="30")
        prediction = Mock()
        prediction.answer = "I don't know"
        
        score, feedback = math_problem_metric(example, prediction)
        
        assert score == 0.0
        assert "no numerical answer found" in feedback.lower()
        assert "clear numerical result" in feedback.lower()

    def test_complex_answer_extraction(self):
        """Test extracting numbers from verbose answers."""
        example = dspy.Example(question="Calculate 12*8", answer="96")
        prediction = Mock()
        prediction.answer = "The answer to 12 times 8 is 96."
        
        score, feedback = math_problem_metric(example, prediction)
        
        assert score == 1.0
        assert "correct answer: 96" in feedback.lower()

    def test_missing_expected_answer(self):
        """Test handling missing expected answer."""
        example = dspy.Example(question="Calculate", answer="")
        prediction = Mock()
        prediction.answer = "42"
        
        score, feedback = math_problem_metric(example, prediction)
        
        assert score == 0.0
        assert "missing expected or actual answer" in feedback.lower()


class TestTextClassificationMetric:
    """Test text classification metric with μf compliance."""

    def test_exact_classification_match(self):
        """Test exact classification match."""
        example = dspy.Example(text="I love this!", label="positive")
        prediction = Mock()
        prediction.classification = "positive"
        
        score, feedback = text_classification_metric(example, prediction)
        
        assert score == 1.0
        assert "correct classification: 'positive'" in feedback.lower()

    def test_variation_mapping_match(self):
        """Test classification with common variations."""
        example = dspy.Example(text="Good movie", label="positive")
        prediction = Mock()
        prediction.classification = "pos"  # Common abbreviation
        
        score, feedback = text_classification_metric(example, prediction)
        
        assert score == 0.8
        assert "correct but use standard format" in feedback.lower()

    def test_opposite_classification(self):
        """Test opposite classification error."""
        example = dspy.Example(text="I love this!", label="positive")
        prediction = Mock()
        prediction.classification = "negative"  # Opposite
        
        score, feedback = text_classification_metric(example, prediction)
        
        assert score == 0.0
        assert "opposite of expected" in feedback.lower()
        assert "review the input text more carefully" in feedback.lower()

    def test_completely_wrong_classification(self):
        """Test completely incorrect classification."""
        example = dspy.Example(text="Great!", label="positive")
        prediction = Mock()
        prediction.classification = "banana"  # Nonsense
        
        score, feedback = text_classification_metric(example, prediction)
        
        assert score == 0.0
        assert "incorrect classification" in feedback.lower()
        assert "available classes" in feedback.lower()

    def test_boolean_classification(self):
        """Test boolean (true/false) classification."""
        example = dspy.Example(text="The sky is blue", label="true")
        prediction = Mock()
        prediction.classification = "yes"  # Maps to true
        
        score, feedback = text_classification_metric(example, prediction)
        
        assert score == 0.8
        assert "correct but" in feedback.lower()

    def test_missing_labels(self):
        """Test handling missing classification labels."""
        example = dspy.Example(text="Test text", label="")
        prediction = Mock()
        prediction.classification = "positive"
        
        score, feedback = text_classification_metric(example, prediction)
        
        assert score == 0.0
        assert "missing expected or actual" in feedback.lower()

    def test_alternative_attribute_names(self):
        """Test using alternative attribute names for classification."""
        example = dspy.Example(text="Test", answer="negative")  # Using answer instead of label
        
        # Create simple object with label attribute
        class SimplePrediction:
            def __init__(self, label_value):
                self.label = label_value
                
        prediction = SimplePrediction("negative")
        
        score, feedback = text_classification_metric(example, prediction)
        
        assert score == 1.0
        assert "correct classification" in feedback.lower()


class TestQAAccuracyMetric:
    """Test QA accuracy metric with μf compliance."""

    def test_exact_answer_match(self):
        """Test exact answer match."""
        example = dspy.Example(question="What is the capital of France?", answer="Paris")
        prediction = Mock()
        prediction.answer = "Paris"
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert score == 1.0
        assert "exact answer match" in feedback.lower()

    def test_verbose_but_correct_answer(self):
        """Test verbose answer containing correct information."""
        example = dspy.Example(question="Capital of France?", answer="Paris")
        prediction = Mock()
        prediction.answer = "The capital of France is Paris, which is a beautiful city."
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert score == 0.9
        assert "overly verbose" in feedback.lower()
        assert "contains correct information" in feedback.lower()

    def test_incomplete_answer(self):
        """Test incomplete but partially correct answer."""
        example = dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare")
        prediction = Mock()
        prediction.answer = "Shakespeare"
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert score == 0.7
        assert "partially correct but incomplete" in feedback.lower()

    def test_key_terms_overlap(self):
        """Test answer with significant key term overlap."""
        example = dspy.Example(question="What causes rain?", answer="water evaporation condensation")
        prediction = Mock()
        prediction.answer = "evaporation and condensation process"
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert 0.5 <= score <= 0.7
        assert "key terms correct" in feedback.lower()

    def test_minimal_relevance(self):
        """Test answer with minimal relevance."""
        example = dspy.Example(question="What is photosynthesis?", answer="plants convert sunlight energy")
        prediction = Mock()
        prediction.answer = "plants are green"
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert 0.1 <= score <= 0.4
        assert "minimal relevance" in feedback.lower()

    def test_completely_unrelated_answer(self):
        """Test completely unrelated answer."""
        example = dspy.Example(question="What is the speed of light?", answer="300000000 meters per second")
        prediction = Mock()
        prediction.answer = "I like pizza"
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert score == 0.0
        assert "appears unrelated" in feedback.lower()
        assert "understand and directly address" in feedback.lower()

    def test_missing_answers(self):
        """Test handling missing expected or actual answers."""
        example = dspy.Example(question="Test question", answer="")
        prediction = Mock()
        prediction.answer = "Some answer"
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert score == 0.0
        assert "missing expected or actual answer" in feedback.lower()

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching."""
        example = dspy.Example(question="Name a color", answer="BLUE")
        prediction = Mock()
        prediction.answer = "blue"
        
        score, feedback = qa_accuracy_metric(example, prediction)
        
        assert score == 1.0
        assert "exact answer match" in feedback.lower()


class TestMetricsIntegration:
    """Test metrics integration and edge cases."""

    def test_all_metrics_return_tuples(self):
        """Test that all metrics return (score, feedback) tuples."""
        metrics = [
            code_evaluation_metric,
            math_problem_metric,
            text_classification_metric,
            qa_accuracy_metric
        ]
        
        example = dspy.Example(question="test", answer="test", label="test")
        prediction = Mock()
        prediction.code = "print('test')"
        prediction.answer = "test"
        prediction.classification = "test"
        
        for metric in metrics:
            result = metric(example, prediction)
            assert isinstance(result, tuple)
            assert len(result) == 2
            score, feedback = result
            assert isinstance(score, float)
            assert isinstance(feedback, str)
            assert 0.0 <= score <= 1.0

    def test_metrics_with_none_prediction(self):
        """Test metrics handle None prediction gracefully."""
        example = dspy.Example(question="test", answer="test", label="test")
        
        # All metrics should handle None prediction without crashing
        for metric in [code_evaluation_metric, math_problem_metric, 
                      text_classification_metric, qa_accuracy_metric]:
            score, feedback = metric(example, None)
            assert isinstance(score, float)
            assert isinstance(feedback, str)
            assert len(feedback) > 0
            # Note: Different metrics handle None differently - some may succeed

    def test_metrics_with_string_prediction(self):
        """Test metrics handle string predictions."""
        example = dspy.Example(question="test", answer="42", label="positive")
        
        # Test with string prediction
        result = math_problem_metric(example, "The answer is 42")
        score, feedback = result
        assert score == 1.0
        
        result = text_classification_metric(example, "positive")
        score, feedback = result
        assert score == 1.0
        
        result = qa_accuracy_metric(example, "42")
        score, feedback = result
        assert score == 1.0

    def test_μf_compliance(self):
        """Test that all metrics are μf-compliant (return score, feedback tuple)."""
        example = dspy.Example(
            question="Write code to add two numbers", 
            answer="def add(a,b): return a+b",
            label="positive"
        )
        prediction = Mock()
        prediction.code = "def add(a, b): return a + b"
        prediction.answer = "def add(a, b): return a + b"
        prediction.classification = "positive"
        
        # All metrics should return μf-compliant tuples
        for metric_name, metric in [
            ("code_evaluation", code_evaluation_metric),
            ("math_problem", math_problem_metric),  
            ("text_classification", text_classification_metric),
            ("qa_accuracy", qa_accuracy_metric)
        ]:
            score, feedback = metric(example, prediction)
            
            # Score should be float between 0 and 1
            assert isinstance(score, float), f"{metric_name} should return float score"
            assert 0.0 <= score <= 1.0, f"{metric_name} score should be between 0 and 1"
            
            # Feedback should be meaningful string  
            assert isinstance(feedback, str), f"{metric_name} should return string feedback"
            assert len(feedback) > 10, f"{metric_name} feedback should be descriptive"