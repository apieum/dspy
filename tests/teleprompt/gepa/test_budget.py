"""Test budget implementations and magic methods."""

import pytest
import dspy
from dspy.teleprompt.gepa.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.gepa.budget.iterations import IterationBudget
from dspy.teleprompt.gepa.budget.adaptive import AdaptiveBudget


class TestBudgetInterface:
    """Test that all budget implementations provide required interface methods."""

    @pytest.mark.parametrize("budget_class,init_value", [
        (LMCallsBudget, 100),
        (IterationBudget, 5),
        (AdaptiveBudget, 100)
    ])
    def test_budget_interface_methods(self, budget_class, init_value):
        """Test all budget types implement required interface methods."""
        budget = budget_class(init_value)

        # Core interface methods
        assert hasattr(budget, 'spend_on_evaluation')
        assert hasattr(budget, 'spend_on_generation')
        assert hasattr(budget, 'spend_on_selection')
        assert hasattr(budget, 'get_remaining')

        # Magic methods for comparisons and type conversion
        assert hasattr(budget, '__gt__')
        assert hasattr(budget, '__ge__')
        assert hasattr(budget, '__lt__')
        assert hasattr(budget, '__le__')
        assert hasattr(budget, '__eq__')
        assert hasattr(budget, '__ne__')
        assert hasattr(budget, '__float__')
        assert hasattr(budget, '__int__')


class TestBudgetTypeConversion:
    """Test budget type conversion and remaining budget calculation."""

    def test_lm_calls_budget_conversion(self):
        """Test LMCallsBudget type conversion and remaining calculation."""
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
        """Test IterationBudget type conversion and remaining calculation."""
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
        """Test AdaptiveBudget type conversion and remaining calculation."""
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


class TestBudgetComparisons:
    """Test budget comparison operators with different numeric types."""

    def test_comparison_operators_with_integers(self):
        """Test all comparison operators work correctly with integers."""
        budget = LMCallsBudget(100)
        budget.consumed_calls = 30  # 70 remaining

        # Test all comparison operators
        assert budget > 50
        assert budget >= 70
        assert budget >= 50
        assert not (budget < 50)
        assert budget <= 70
        assert not (budget <= 50)
        assert budget == 70
        assert budget != 50
        assert budget != 100

    def test_comparison_operators_with_floats(self):
        """Test all comparison operators work correctly with floats."""
        budget = IterationBudget(10)
        budget.current_iteration = 3  # 7 remaining

        # Test all comparison operators with floats
        assert budget > 5.0
        assert budget >= 7.0
        assert budget >= 5.5
        assert not (budget < 5.0)
        assert budget <= 7.0
        assert not (budget <= 5.0)
        assert budget == 7.0
        assert budget != 5.0
        assert budget != 10.0

    def test_edge_cases_and_boundary_conditions(self):
        """Test budget comparison edge cases and boundary conditions."""
        budget = LMCallsBudget(10)

        # Test zero budget
        budget.consumed_calls = 10
        assert budget == 0
        assert budget <= 0
        assert not (budget > 0)
        assert budget >= 0
        assert not (budget < 0)

        # Test float precision
        assert budget == 0.0
        assert not (budget > 0.1)

        # Test negative comparisons (budget can't go negative, but comparison should work)
        budget = AdaptiveBudget(100)
        budget.consumed_budget = 25  # 75 remaining
        assert budget > 74.9
        assert budget < 76
        assert budget == 75.0

    def test_unsupported_comparison_types(self):
        """Test that unsupported comparison types return NotImplemented."""
        budget = LMCallsBudget(100)

        # Test comparison with unsupported types returns NotImplemented
        assert budget.__gt__("string") is NotImplemented
        assert budget.__eq__([1, 2, 3]) is NotImplemented
        assert budget.__lt__({"key": "value"}) is NotImplemented


class TestBudgetSpending:
    """Test budget spending methods for different budget types."""

    def test_lm_calls_budget_spending(self):
        """Test LMCallsBudget spending methods work correctly."""
        budget = LMCallsBudget(100)

        # Create a module with simulated LLM call history
        module = dspy.Predict("input -> output")
        module.history = ["call1", "call2"]  # Simulate 2 LLM calls

        # Test evaluation spending (tracks actual LLM calls from history)
        initial_calls = budget.consumed_calls
        budget.spend_on_evaluation(module, {"phase": "evaluation"})
        assert budget.consumed_calls == initial_calls + len(module.history)

        # Test generation spending (adds standard generation cost)
        initial_calls = budget.consumed_calls
        budget.spend_on_generation(module, {"type": "mutation"})
        assert budget.consumed_calls == initial_calls + 1  # Standard generation cost

        # Test selection spending (no-op for LMCallsBudget)
        initial_calls = budget.consumed_calls
        budget.spend_on_selection(5, 2, {"phase": "selection"})
        assert budget.consumed_calls == initial_calls  # No change

    def test_spending_with_edge_cases(self):
        """Test spending methods handle edge cases gracefully."""
        budget = LMCallsBudget(100)

        # Test with module without history
        module = dspy.Predict("input -> output")
        budget.spend_on_evaluation(module)
        assert budget.consumed_calls == 0  # No calls to track

        budget.spend_on_generation(module)
        assert budget.consumed_calls == 1  # Standard generation cost

        # Test with None module
        budget.spend_on_evaluation(None)
        assert budget.consumed_calls == 1  # No change

        budget.spend_on_generation(None)
        assert budget.consumed_calls == 2  # Standard generation cost

        budget.spend_on_selection(0, 0)
        assert budget.consumed_calls == 2  # No change

    def test_spending_metadata_parameter(self):
        """Test that spending methods accept and handle metadata parameter."""
        budget = LMCallsBudget(100)
        module = dspy.Predict("input -> output")
        module.history = ["call1"]

        # Test spending with various metadata
        budget.spend_on_evaluation(module, {"phase": "minibatch", "size": 5})
        budget.spend_on_generation(module, {"strategy": "reflection", "mutations": 3})
        budget.spend_on_selection(10, 3, {"algorithm": "pareto_frontier"})

        # Should not raise exceptions and should track calls appropriately
        assert budget.consumed_calls == 2  # 1 evaluation + 1 generation


class TestBudgetIntegration:
    """Integration tests for budget functionality in GEPA context."""

    def test_budget_exhaustion_detection(self):
        """Test that budget exhaustion is properly detected."""
        budget = LMCallsBudget(5)

        # Consume most of the budget
        budget.consumed_calls = 4
        assert budget > 0
        assert int(budget) == 1

        # Exhaust the budget completely
        budget.consumed_calls = 5
        assert budget <= 0
        assert int(budget) == 0
        assert not (budget > 0)

    def test_different_budget_types_behavior(self):
        """Test that different budget types behave consistently."""
        llm_budget = LMCallsBudget(100)
        iter_budget = IterationBudget(10)
        adaptive_budget = AdaptiveBudget(50)

        # All should support the same interface
        for budget in [llm_budget, iter_budget, adaptive_budget]:
            assert budget > 0
            assert int(budget) > 0
            assert float(budget) > 0.0
            assert budget == int(budget)
            assert budget.get_remaining()
