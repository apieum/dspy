"""Test budget magic methods for type conversion and comparisons."""

import pytest
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget
from dspy.teleprompt.gepa.budget.iterations import IterationBudget
from dspy.teleprompt.gepa.budget.adaptive import AdaptiveBudget


class TestBudgetMagicMethods:
    """Test budget magic methods for type conversion and comparisons."""

    def test_llm_calls_budget_conversion(self):
        """Test LLMCallsBudget type conversion methods."""
        budget = LLMCallsBudget(100)
        
        # Test initial state
        assert int(budget) == 100
        assert float(budget) == 100.0
        assert budget.get_remaining()["calls"] == 100
        
        # Test after consuming budget
        budget.consumed_calls = 30
        assert int(budget) == 70
        assert float(budget) == 70.0
        assert budget.get_remaining()["calls"] == 70

    def test_iteration_budget_conversion(self):
        """Test IterationBudget type conversion methods."""
        budget = IterationBudget(10)
        
        # Test initial state
        assert int(budget) == 10
        assert float(budget) == 10.0
        assert budget.get_remaining()["iterations"] == 10
        
        # Test after consuming iterations
        budget.current_iteration = 3
        assert int(budget) == 7
        assert float(budget) == 7.0
        assert budget.get_remaining()["iterations"] == 7

    def test_adaptive_budget_conversion(self):
        """Test AdaptiveBudget type conversion methods."""
        budget = AdaptiveBudget(50)
        
        # Test initial state
        assert int(budget) == 50
        assert float(budget) == 50.0
        assert budget.get_remaining()["budget"] == 50
        
        # Test after consuming budget
        budget.consumed_budget = 20
        assert int(budget) == 30
        assert float(budget) == 30.0
        assert budget.get_remaining()["budget"] == 30

    def test_budget_comparisons_int(self):
        """Test budget comparisons with integers."""
        budget = LLMCallsBudget(100)
        
        # Test all comparison operators with int
        assert budget > 50
        assert budget >= 100
        assert budget >= 50
        assert not (budget < 50)
        assert budget <= 100
        assert not (budget <= 50)
        assert budget == 100
        assert budget != 50
        
        # Test after consuming budget
        budget.consumed_calls = 30
        assert budget > 50
        assert budget >= 70
        assert not (budget >= 80)
        assert budget < 80
        assert budget <= 70
        assert budget <= 80
        assert budget == 70
        assert budget != 100

    def test_budget_comparisons_float(self):
        """Test budget comparisons with floats."""
        budget = IterationBudget(10)
        
        # Test all comparison operators with float
        assert budget > 5.0
        assert budget >= 10.0
        assert budget >= 5.5
        assert not (budget < 5.0)
        assert budget <= 10.0
        assert not (budget <= 5.0)
        assert budget == 10.0
        assert budget != 5.0
        
        # Test after consuming iterations
        budget.current_iteration = 3
        assert budget > 5.0
        assert budget >= 7.0
        assert not (budget >= 8.0)
        assert budget < 8.0
        assert budget <= 7.0
        assert budget <= 8.0
        assert budget == 7.0
        assert budget != 10.0

    def test_budget_edge_cases(self):
        """Test budget edge cases and boundary conditions."""
        budget = LLMCallsBudget(10)
        
        # Test zero budget
        budget.consumed_calls = 10
        assert budget == 0
        assert budget <= 0
        assert not (budget > 0)
        assert budget >= 0
        
        # Test negative comparisons (should work even if budget can't go negative)
        assert not (budget < 0)
        assert budget >= 0
        
        # Test float precision
        assert budget == 0.0
        assert not (budget > 0.1)

    def test_dynamic_type_conversion(self):
        """Test that type(other)(self) pattern works correctly."""
        budget = AdaptiveBudget(100)
        
        # Test that comparisons use the right type conversion
        # This verifies type(other)(self) pattern is working
        
        # Integer comparison should use int(budget)
        assert budget > 50  # Uses int(budget) > 50
        
        # Float comparison should use float(budget)  
        assert budget > 50.5  # Uses float(budget) > 50.5
        
        # Mixed precision comparisons
        budget.consumed_budget = 25
        assert budget == 75    # int comparison
        assert budget == 75.0  # float comparison
        assert budget > 74.9   # float comparison
        assert budget < 76     # int comparison

    def test_not_implemented_comparisons(self):
        """Test that unsupported comparison types return NotImplemented."""
        budget = LLMCallsBudget(100)
        
        # Test comparison with unsupported types
        result = budget.__gt__("string")
        assert result is NotImplemented
        
        result = budget.__eq__([1, 2, 3])
        assert result is NotImplemented
        
        result = budget.__lt__({"key": "value"})
        assert result is NotImplemented

    def test_budget_spending_methods(self):
        """Test that spending methods work and can be tracked."""
        budget = LLMCallsBudget(100)
        
        # Create a mock module for testing
        class MockModule:
            def __init__(self):
                self.history = ["call1", "call2", "call3"]
        
        module = MockModule()
        
        # Test spend_on_evaluation
        initial_calls = budget.consumed_calls
        budget.spend_on_evaluation(module, {"phase": "test"})
        assert budget.consumed_calls > initial_calls
        
        # Test spend_on_generation  
        initial_calls = budget.consumed_calls
        budget.spend_on_generation(module, {"type": "mutation"})
        assert budget.consumed_calls > initial_calls
        
        # Test spend_on_selection (should be no-op in base implementation)
        initial_calls = budget.consumed_calls
        budget.spend_on_selection(10, 3, {"strategy": "pareto"})
        # For LLMCallsBudget, selection doesn't consume LLM calls
        assert budget.consumed_calls == initial_calls