"""Test budget implementations and magic methods."""

import dspy
from dspy.teleprompt.gepa.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.gepa.budget.iterations import IterationBudget
from dspy.teleprompt.gepa.budget.adaptive import AdaptiveBudget


class TestBudgetInterface:
    """Test budget new interface methods."""

    def test_lm_calls_budget_interface(self):
        """Test LMCallsBudget implements new interface."""
        budget = LMCallsBudget(100)

        # New interface methods
        assert hasattr(budget, 'spend_on_evaluation')
        assert hasattr(budget, 'spend_on_generation')
        assert hasattr(budget, 'spend_on_selection')
        assert hasattr(budget, 'get_remaining')

        # Magic methods
        assert hasattr(budget, '__gt__')
        assert hasattr(budget, '__float__')
        assert hasattr(budget, '__int__')

    def test_iteration_budget_interface(self):
        """Test IterationBudget implements new interface."""
        budget = IterationBudget(5)

        assert hasattr(budget, 'spend_on_evaluation')
        assert hasattr(budget, 'spend_on_generation')
        assert hasattr(budget, 'spend_on_selection')
        assert hasattr(budget, 'get_remaining')

        # Magic methods
        assert hasattr(budget, '__gt__')
        assert hasattr(budget, '__float__')

    def test_adaptive_budget_interface(self):
        """Test AdaptiveBudget implements new interface."""
        budget = AdaptiveBudget(100)

        assert hasattr(budget, 'spend_on_evaluation')
        assert hasattr(budget, 'spend_on_generation')
        assert hasattr(budget, 'spend_on_selection')
        assert hasattr(budget, 'get_remaining')

        # Magic methods
        assert hasattr(budget, '__gt__')
        assert hasattr(budget, '__float__')


class TestBudgetMagicMethods:
    """Test budget magic comparison methods."""

    def test_lm_calls_budget_conversion(self):
        """Test LMCallsBudget type conversion methods."""
        budget = LMCallsBudget(100)
        budget.consumed_calls = 30  # 70 remaining

        # Type conversion methods
        assert float(budget) == 70.0
        assert int(budget) == 70

    def test_iteration_budget_conversion(self):
        """Test IterationBudget type conversion methods."""
        budget = IterationBudget(5)
        budget.current_iteration = 2  # 3 remaining

        assert float(budget) == 3.0
        assert int(budget) == 3

    def test_adaptive_budget_conversion(self):
        """Test AdaptiveBudget type conversion methods."""
        budget = AdaptiveBudget(100)
        budget.consumed_budget = 25  # 75 remaining

        assert float(budget) == 75.0
        assert int(budget) == 75

    def test_budget_comparisons_int(self):
        """Test budget comparison operators with integers."""
        budget = LMCallsBudget(100)
        budget.consumed_calls = 30  # 70 remaining

        # Comparison operators
        assert budget > 50
        assert budget >= 70
        assert budget == 70
        assert budget != 50
        assert budget <= 70
        assert not (budget < 70)

    def test_budget_comparisons_float(self):
        """Test budget comparison operators with floats."""
        budget = LMCallsBudget(100)
        budget.consumed_calls = 25  # 75 remaining

        assert budget > 50.5
        assert budget >= 75.0
        assert budget == 75.0
        assert budget != 74.9
        assert budget <= 75.0
        assert not (budget < 75.0)

    def test_budget_edge_cases(self):
        """Test budget comparison edge cases."""
        # Empty budget
        empty_budget = LMCallsBudget(50)
        empty_budget.consumed_calls = 50

        assert empty_budget == 0
        assert empty_budget <= 0
        assert not (empty_budget > 0)

        # Full budget
        full_budget = LMCallsBudget(100)

        assert full_budget == 100
        assert full_budget > 50
        assert full_budget >= 100

    def test_dynamic_type_conversion(self):
        """Test dynamic type conversion pattern."""
        budget = LMCallsBudget(100)
        budget.consumed_calls = 20  # 80 remaining

        # Should work with different numeric types
        assert budget > 79
        assert budget > 79.5
        assert budget == 80
        assert budget == 80.0

    def test_not_implemented_comparisons(self):
        """Test comparisons return NotImplemented for unsupported types."""
        budget = LMCallsBudget(100)

        # Should return NotImplemented (which Python handles gracefully)
        result = budget.__gt__("string")
        assert result is NotImplemented

        result = budget.__eq__([1, 2, 3])
        assert result is NotImplemented


class TestBudgetSpending:
    """Test budget spending methods."""

    def test_budget_spending_methods(self):
        """Test budget spending methods work correctly."""
        budget = LMCallsBudget(100)

        # Create a mock module with history
        module = dspy.Predict("input -> output")
        module.history = ["call1", "call2"]  # Simulate 2 LLM calls

        # Test evaluation spending
        budget.spend_on_evaluation(module, {"phase": "evaluation"})
        assert budget.consumed_calls == 2

        # Test generation spending
        budget.spend_on_generation(module, {"type": "mutation"})
        assert budget.consumed_calls == 3  # 2 + 1 standard generation cost

        # Test selection spending (no-op for LMCallsBudget)
        budget.spend_on_selection(5, 2, {"phase": "selection"})
        assert budget.consumed_calls == 3  # No change

    def test_spending_with_empty_module(self):
        """Test spending methods handle modules without history."""
        budget = LMCallsBudget(100)

        module = dspy.Predict("input -> output")
        # No history set

        budget.spend_on_evaluation(module)
        assert budget.consumed_calls == 0  # No calls to track

        budget.spend_on_generation(module)
        assert budget.consumed_calls == 1  # Standard generation cost

    def test_spending_with_none_module(self):
        """Test spending methods handle None module gracefully."""
        budget = LMCallsBudget(100)

        budget.spend_on_evaluation(None)
        assert budget.consumed_calls == 0

        budget.spend_on_generation(None)
        assert budget.consumed_calls == 1  # Standard generation cost

        budget.spend_on_selection(0, 0)
        assert budget.consumed_calls == 1  # No change"""Test budget magic methods for type conversion and comparisons."""

import pytest
from dspy.teleprompt.gepa.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.gepa.budget.iterations import IterationBudget
from dspy.teleprompt.gepa.budget.adaptive import AdaptiveBudget


class TestBudgetMagicMethods:
    """Test budget magic methods for type conversion and comparisons."""

    def test_lm_calls_budget_conversion(self):
        """Test LMCallsBudget type conversion methods."""
        budget = LMCallsBudget(100)

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
        budget = LMCallsBudget(100)

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
        budget = LMCallsBudget(10)

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
        budget = LMCallsBudget(100)

        # Test comparison with unsupported types
        result = budget.__gt__("string")
        assert result is NotImplemented

        result = budget.__eq__([1, 2, 3])
        assert result is NotImplemented

        result = budget.__lt__({"key": "value"})
        assert result is NotImplemented

    def test_budget_spending_methods(self):
        """Test that spending methods work and can be tracked."""
        budget = LMCallsBudget(100)

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
        # For LMCallsBudget, selection doesn't consume LLM calls
        assert budget.consumed_calls == initial_calls
