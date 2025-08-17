"""Performance tests to ensure Darwin GEPA is reasonably efficient."""

import time
import dspy
from dspy.teleprompt.darwin import GEPAMute
from dspy.utils.dummies import DummyLM


class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


def fast_metric(example, prediction, trace=None):
    """Fast metric for performance testing."""
    return 1.0 if hasattr(prediction, 'answer') else 0.0


class TestPerformance:
    """Test performance characteristics of Darwin GEPA."""

    def test_optimization_speed(self):
        """Test that optimization completes in reasonable time."""
        # Medium-sized dataset for performance testing
        trainset = [
            dspy.Example(question=f"Speed test question {i}", answer=f"answer{i}").with_inputs("question")
            for i in range(10)
        ]
        
        # Provide sufficient responses
        responses = [{"answer": f"answer{i}"} for i in range(10)]
        responses.append({"response": "Fast optimization response."})
        responses.extend([{"answer": f"optimized{i}"} for i in range(5)])
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(fast_metric, max_calls=8, patience=2)
            
            start_time = time.time()
            result = optimizer.compile(student, trainset=trainset)
            end_time = time.time()
            
            # Should complete quickly (under 5 seconds for this test size)
            optimization_time = end_time - start_time
            assert optimization_time < 5.0, f"Optimization took {optimization_time:.2f}s, should be under 5s"
            assert result._compiled is True

    def test_memory_efficiency(self):
        """Test that optimization doesn't accumulate excessive memory."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        trainset = [
            dspy.Example(question=f"Memory test {i}", answer=f"answer{i}").with_inputs("question")
            for i in range(15)
        ]
        
        responses = [{"answer": f"answer{i}"} for i in range(15)]
        responses.append({"response": "Memory-efficient optimization."})
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(fast_metric, max_calls=6, patience=2)
            
            result = optimizer.compile(student, trainset=trainset)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage after optimization
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Should not increase memory by more than 50MB for this test
            assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, should be under 50MB"
            assert result._compiled is True

    def test_budget_efficiency(self):
        """Test that optimization uses budget efficiently."""
        trainset = [
            dspy.Example(question="Budget efficiency test", answer="efficient").with_inputs("question"),
            dspy.Example(question="Another test", answer="efficient2").with_inputs("question"),
        ]
        
        responses = [
            {"answer": "efficient"},
            {"answer": "efficient2"},
            {"response": "Budget-efficient optimization."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(fast_metric, max_calls=5, patience=2)
            
            result = optimizer.compile(student, trainset=trainset)
            
            # Check that budget was used reasonably
            budget = optimizer.budget
            # Budget may be slightly exceeded due to evaluation phases, but should be reasonable
            assert budget.consumed_calls <= budget.max_calls * 1.5  # Allow some overage
            assert budget.consumed_calls > 0  # Should have used some budget
            assert result._compiled is True

    def test_scalability_with_data_size(self):
        """Test that performance scales reasonably with data size."""
        def measure_optimization_time(dataset_size):
            trainset = [
                dspy.Example(question=f"Scale test {i}", answer=f"answer{i}").with_inputs("question")
                for i in range(dataset_size)
            ]
            
            responses = [{"answer": f"answer{i}"} for i in range(dataset_size)]
            responses.append({"response": "Scalable optimization."})
            
            dummy_lm = DummyLM(responses)
            
            with dspy.context(lm=dummy_lm):
                student = SimpleQA()
                optimizer = GEPAMute(fast_metric, max_calls=min(dataset_size + 2, 10), patience=2)
                
                start_time = time.time()
                result = optimizer.compile(student, trainset=trainset)
                end_time = time.time()
                
                return end_time - start_time, result
        
        # Test with small and medium datasets
        small_time, small_result = measure_optimization_time(3)
        medium_time, medium_result = measure_optimization_time(8)
        
        # Performance should scale reasonably (not exponentially)
        # Allow some flexibility but shouldn't be dramatically slower
        assert medium_time < small_time * 5, f"Medium dataset took {medium_time:.2f}s vs small {small_time:.2f}s"
        assert small_result._compiled is True
        assert medium_result._compiled is True

    def test_repeated_optimization_consistency(self):
        """Test that repeated optimizations are consistent in performance."""
        trainset = [
            dspy.Example(question="Consistency test", answer="consistent").with_inputs("question"),
        ]
        
        responses = [
            {"answer": "consistent"},
            {"response": "Consistent optimization."},
        ]
        
        times = []
        
        for i in range(3):  # Run multiple times
            dummy_lm = DummyLM(responses.copy())  # Fresh LM each time
            
            with dspy.context(lm=dummy_lm):
                student = SimpleQA()
                optimizer = GEPAMute(fast_metric, max_calls=3, patience=1)
                
                start_time = time.time()
                result = optimizer.compile(student, trainset=trainset)
                end_time = time.time()
                
                times.append(end_time - start_time)
                assert result._compiled is True
        
        # Times should be reasonably consistent (within 2x of each other)
        min_time = min(times)
        max_time = max(times)
        assert max_time < min_time * 3, f"Optimization times vary too much: {times}"

    def test_low_latency_compilation(self):
        """Test that minimal compilations have low latency."""
        # Single example for minimal compilation
        trainset = [dspy.Example(question="Quick test", answer="quick").with_inputs("question")]
        
        responses = [
            {"answer": "quick"},
            {"response": "Quick optimization."},
        ]
        
        dummy_lm = DummyLM(responses)
        
        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            optimizer = GEPAMute(fast_metric, max_calls=2, patience=1)
            
            start_time = time.time()
            result = optimizer.compile(student, trainset=trainset)
            end_time = time.time()
            
            # Minimal compilation should be very fast (under 1 second)
            compilation_time = end_time - start_time
            assert compilation_time < 1.0, f"Minimal compilation took {compilation_time:.2f}s, should be under 1s"
            assert result._compiled is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])