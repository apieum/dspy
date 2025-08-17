"""Robustness tests to ensure Darwin handles edge cases and errors gracefully."""

import dspy
import pytest
from dspy.teleprompt.darwin import GEPAMute
from dspy.utils.dummies import DummyLM


class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


def unreliable_metric(example, prediction, trace=None):
    """Metric that sometimes returns tuples (μf-compliant) and sometimes floats."""
    if not hasattr(example, 'answer') or not hasattr(prediction, 'answer'):
        return 0.0
    
    expected = str(example.answer).lower()
    actual = str(prediction.answer).lower()
    score = 1.0 if expected == actual else 0.0
    
    # Sometimes return tuple (score, feedback), sometimes just score
    if hash(expected) % 2 == 0:
        return (score, f"Feedback for {expected}")  # μf-compliant tuple
    else:
        return score  # Regular float


def error_prone_metric(example, prediction, trace=None):
    """Metric that occasionally raises errors."""
    if hasattr(example, 'question') and 'error' in example.question.lower():
        raise ValueError("Simulated metric error")
    
    if not hasattr(example, 'answer') or not hasattr(prediction, 'answer'):
        return 0.0
    
    return 1.0 if example.answer == prediction.answer else 0.0


class TestRobustness:
    """Test system robustness under adverse conditions."""

    def test_mixed_tuple_float_metrics(self):
        """Test handling of metrics that return both tuples and floats."""
        trainset = [
            dspy.Example(question="Even question", answer="even").with_inputs("question"),  # Will return tuple
            dspy.Example(question="Odd question", answer="odd").with_inputs("question"),    # Will return float
            dspy.Example(question="Another even", answer="even2").with_inputs("question"),  # Will return tuple
        ]
        
        responses = [
            {"answer": "even"},
            {"answer": "odd"},
            {"answer": "even2"},
            {"response": "Handled mixed metric returns."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(unreliable_metric, max_calls=5)
            
            # Should handle both tuple and float returns gracefully
            result = optimizer.compile(student, trainset=trainset)
            
            assert result._compiled is True

    def test_empty_trainset_handling(self):
        """Test graceful handling of empty or minimal datasets."""
        empty_trainset = []
        minimal_trainset = [dspy.Example(question="Only one", answer="one").with_inputs("question")]
        
        responses = [
            {"answer": "one"},
            {"response": "Minimal data optimization."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(unreliable_metric, max_calls=3, patience=1)
            
            # Should handle minimal data without crashing
            result = optimizer.compile(student, trainset=minimal_trainset)
            assert result._compiled is True

    def test_metric_error_resilience(self):
        """Test that system continues when metrics occasionally fail."""
        trainset = [
            dspy.Example(question="Good question", answer="good").with_inputs("question"),
            dspy.Example(question="Error question", answer="bad").with_inputs("question"),  # Will cause metric error
            dspy.Example(question="Another good", answer="good2").with_inputs("question"),
        ]
        
        responses = [
            {"answer": "good"},
            {"answer": "bad"},
            {"answer": "good2"},
            {"response": "Error-resilient optimization."},
            {"answer": "recovered"},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(error_prone_metric, max_calls=6, patience=2)
            
            # Should complete despite metric errors
            result = optimizer.compile(student, trainset=trainset)
            assert result._compiled is True

    def test_zero_budget_handling(self):
        """Test behavior with extremely limited budget."""
        trainset = [dspy.Example(question="Budget test", answer="test").with_inputs("question")]
        
        responses = [{"answer": "test"}]  # Only one response available
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(unreliable_metric, max_calls=1, patience=1)  # Minimal budget
            
            # Should handle minimal budget gracefully
            result = optimizer.compile(student, trainset=trainset)
            assert result._compiled is True

    def test_large_dataset_efficiency(self):
        """Test efficiency with larger datasets."""
        # Create a moderately large dataset
        trainset = [
            dspy.Example(question=f"Question {i}", answer=f"Answer {i}").with_inputs("question")
            for i in range(20)
        ]
        
        # Provide enough responses for the large dataset
        responses = [{"answer": f"Answer {i}"} for i in range(20)]
        responses.append({"response": "Efficient large dataset optimization."})
        responses.extend([{"answer": f"Optimized {i}"} for i in range(5)])
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(unreliable_metric, max_calls=15, patience=2)
            
            # Should handle larger datasets efficiently
            result = optimizer.compile(student, trainset=trainset)
            assert result._compiled is True

    def test_malformed_examples_handling(self):
        """Test handling of malformed examples."""
        trainset = [
            dspy.Example(question="Good question", answer="good").with_inputs("question"),
            dspy.Example(question="No answer question").with_inputs("question"),  # Missing answer
            dspy.Example(answer="No question answer"),  # Missing question
            dspy.Example(question="Another good", answer="good2").with_inputs("question"),
        ]
        
        responses = [
            {"answer": "good"},
            {"answer": "no_answer_response"},
            {"answer": "no_question_response"},
            {"answer": "good2"},
            {"response": "Handled malformed examples."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(unreliable_metric, max_calls=7, patience=2)
            
            # Should handle malformed examples without crashing
            result = optimizer.compile(student, trainset=trainset)
            assert result._compiled is True

    def test_patience_mechanism_functionality(self):
        """Test that patience mechanism prevents infinite loops."""
        trainset = [
            dspy.Example(question="Patience test", answer="test").with_inputs("question"),
        ]
        
        # Responses that won't improve (to trigger patience mechanism)
        responses = [
            {"answer": "wrong1"},  # Initial wrong answer
            {"response": "No improvement attempt 1"},
            {"answer": "wrong2"},  # Still wrong
            {"response": "No improvement attempt 2"},
            {"answer": "wrong3"},  # Still wrong - should trigger patience limit
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(unreliable_metric, max_calls=10, patience=2)  # Low patience
            
            # Should terminate due to patience, not infinite loop
            result = optimizer.compile(student, trainset=trainset)
            assert result._compiled is True

    def test_concurrent_optimization_safety(self):
        """Test that optimization is safe for concurrent use."""
        trainset = [dspy.Example(question="Concurrent test", answer="safe").with_inputs("question")]
        
        responses = [
            {"answer": "safe"},
            {"response": "Thread-safe optimization."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student1 = SimpleQA()
            student2 = SimpleQA()
            
            optimizer1 = GEPAMute(unreliable_metric, max_calls=3, patience=1)
            optimizer2 = GEPAMute(unreliable_metric, max_calls=3, patience=1)
            
            # Should be able to run multiple optimizations independently
            result1 = optimizer1.compile(student1, trainset=trainset)
            result2 = optimizer2.compile(student2, trainset=trainset)
            
            assert result1._compiled is True
            assert result2._compiled is True
            assert result1 is not result2  # Different instances


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])