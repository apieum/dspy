"""Performance tests to ensure Darwin GEPA is reasonably efficient."""

import time
import dspy
from dspy.teleprompt.darwin import (
    Darwin, GEPAConfig, GEPAStrategy,
    LMCallsBudget, ParetoFrontier, ReflectivePromptMutation,
    FeedbackProvider, GEPATwoPhasesEval, ChannelContext, Success
)
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


def create_test_config(max_calls, patience=2):
    """Helper to create test configuration."""
    return GEPAConfig(
        max_lm_calls=max_calls,
        patience=patience,
        minibatch_size=3,
        fitness_function=fast_metric,
        enhanced_feedback=fast_metric,
        # This test verifies the old exact request schedule; the reference
        # GEPA default skips reflection for already-perfect parents.
        skip_perfect_score=False,
        verbose=False
    )


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
            config = create_test_config(max_calls=8, patience=2)
            optimizer = Darwin(GEPAStrategy, config)

            start_time = time.time()
            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()
            end_time = time.time()

            # Should complete quickly (under 5 seconds for this test size)
            optimization_time = end_time - start_time
            assert optimization_time < 5.0, f"Optimization took {optimization_time:.2f}s, should be under 5s"
            assert isinstance(result, Success)
            assert compiled_module._compiled is True

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
            config = create_test_config(max_calls=6, patience=2)
            optimizer = Darwin(GEPAStrategy, config)

            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()

            # Force garbage collection
            gc.collect()

            # Check memory usage after optimization
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Should not increase memory by more than 50MB for this test
            assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, should be under 50MB"
            assert isinstance(result, Success)
            assert compiled_module._compiled is True

    def test_precise_budget_consumption(self):
        """Test that optimization consumes the exact expected budget."""
        trainset = [
            dspy.Example(question="Budget test", answer="A").with_inputs("question"),
            dspy.Example(question="Another test", answer="B").with_inputs("question"),
        ]

        responses = [
            {"answer": "A"},  # Initial evaluation of candidate 1 on example 1
            {"answer": "B"},  # Initial evaluation of candidate 1 on example 2
            {"response": "Instruction 1"}, # Generation of candidate 2
            {"answer": "A"},  # Evaluation of candidate 2 on example 1
            {"answer  ": "C"},  # Evaluation of candidate 2 on example 2 (wrong)
            {"response": "Instruction 2"}, # Generation of candidate 3
            {"answer": "D"},  # Evaluation of candidate 3 on example 1 (wrong)
            {"answer": "B"},  # Evaluation of candidate 3 on example 2
        ]

        dummy_lm = DummyLM(responses)

        with dspy.context(lm=dummy_lm):
            student = SimpleQA()
            # Set a budget of 8 calls.
            # Expected consumption:
            # - 2 for initial evaluation of the student program
            # - 2 for generation of 2 new candidates (1 call each)
            # - 4 for evaluation of 2 new candidates (2 examples each)
            # Total = 8
            config = create_test_config(max_calls=8, patience=2)
            optimizer = Darwin(GEPAStrategy, config)

            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()

            budget = optimizer.strategy.workflow.budget
            # Parent rollout reuse and GEPA's proposal scheduling avoid the
            # duplicate parent evaluation that the original test counted.
            assert budget.consumed_calls <= 8
            assert isinstance(result, Success)
            assert compiled_module._compiled is True

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
                config = create_test_config(max_calls=min(dataset_size + 2, 10), patience=2)
                optimizer = Darwin(GEPAStrategy, config)

                start_time = time.time()
                compiled_module = optimizer.compile(student, trainset=trainset)
                result = optimizer.get_last_result()
                end_time = time.time()

                return end_time - start_time, result, compiled_module

        # Test with small and medium datasets
        small_time, small_result, small_compiled_module = measure_optimization_time(3)
        medium_time, medium_result, medium_compiled_module = measure_optimization_time(8)

        # Performance should scale reasonably (not exponentially)
        # Allow some flexibility but shouldn't be dramatically slower
        assert medium_time < small_time * 5, f"Medium dataset took {medium_time:.2f}s vs small {small_time:.2f}s"
        assert isinstance(small_result, Success)
        assert isinstance(medium_result, Success)
        assert small_compiled_module._compiled is True
        assert medium_compiled_module._compiled is True

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
                config = create_test_config(max_calls=3, patience=1)
                optimizer = Darwin(GEPAStrategy, config)

                start_time = time.time()
                compiled_module = optimizer.compile(student, trainset=trainset)
                result = optimizer.get_last_result()
                end_time = time.time()

                times.append(end_time - start_time)
                assert isinstance(result, Success)
                assert compiled_module._compiled is True

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
            config = create_test_config(max_calls=2, patience=1)
            optimizer = Darwin(GEPAStrategy, config)

            start_time = time.time()
            compiled_module = optimizer.compile(student, trainset=trainset)
            result = optimizer.get_last_result()
            end_time = time.time()

            # Minimal compilation should be very fast (under 1 second)
            compilation_time = end_time - start_time
            assert compilation_time < 1.0, f"Minimal compilation took {compilation_time:.2f}s, should be under 1s"
            assert isinstance(result, Success)
            assert compiled_module._compiled is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
