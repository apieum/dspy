"""Test budget implementations and magic methods."""

import dspy
from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget
from dspy.teleprompt.gepa.budget.iterations import IterationBudget
from dspy.teleprompt.gepa.budget.adaptive import AdaptiveBudget


class TestBudgetInterface:
    """Test budget new interface methods."""
    
    def test_llm_calls_budget_interface(self):
        """Test LLMCallsBudget implements new interface."""
        budget = LLMCallsBudget(100)
        
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
    
    def test_llm_calls_budget_conversion(self):
        """Test LLMCallsBudget type conversion methods."""
        budget = LLMCallsBudget(100)
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
        budget = LLMCallsBudget(100)
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
        budget = LLMCallsBudget(100)
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
        empty_budget = LLMCallsBudget(50)
        empty_budget.consumed_calls = 50
        
        assert empty_budget == 0
        assert empty_budget <= 0
        assert not (empty_budget > 0)
        
        # Full budget
        full_budget = LLMCallsBudget(100)
        
        assert full_budget == 100
        assert full_budget > 50
        assert full_budget >= 100
    
    def test_dynamic_type_conversion(self):
        """Test dynamic type conversion pattern."""
        budget = LLMCallsBudget(100)
        budget.consumed_calls = 20  # 80 remaining
        
        # Should work with different numeric types
        assert budget > 79
        assert budget > 79.5
        assert budget == 80
        assert budget == 80.0
    
    def test_not_implemented_comparisons(self):
        """Test comparisons return NotImplemented for unsupported types."""
        budget = LLMCallsBudget(100)
        
        # Should return NotImplemented (which Python handles gracefully)
        result = budget.__gt__("string")
        assert result is NotImplemented
        
        result = budget.__eq__([1, 2, 3])
        assert result is NotImplemented


class TestBudgetSpending:
    """Test budget spending methods."""
    
    def test_budget_spending_methods(self):
        """Test budget spending methods work correctly."""
        budget = LLMCallsBudget(100)
        
        # Create a mock module with history
        module = dspy.Predict("input -> output")
        module.history = ["call1", "call2"]  # Simulate 2 LLM calls
        
        # Test evaluation spending
        budget.spend_on_evaluation(module, {"phase": "evaluation"})
        assert budget.consumed_calls == 2
        
        # Test generation spending
        budget.spend_on_generation(module, {"type": "mutation"})
        assert budget.consumed_calls == 3  # 2 + 1 standard generation cost
        
        # Test selection spending (no-op for LLMCallsBudget)
        budget.spend_on_selection(5, 2, {"phase": "selection"})
        assert budget.consumed_calls == 3  # No change
    
    def test_spending_with_empty_module(self):
        """Test spending methods handle modules without history."""
        budget = LLMCallsBudget(100)
        
        module = dspy.Predict("input -> output")
        # No history set
        
        budget.spend_on_evaluation(module)
        assert budget.consumed_calls == 0  # No calls to track
        
        budget.spend_on_generation(module)
        assert budget.consumed_calls == 1  # Standard generation cost
    
    def test_spending_with_none_module(self):
        """Test spending methods handle None module gracefully."""
        budget = LLMCallsBudget(100)
        
        budget.spend_on_evaluation(None)
        assert budget.consumed_calls == 0
        
        budget.spend_on_generation(None)
        assert budget.consumed_calls == 1  # Standard generation cost
        
        budget.spend_on_selection(0, 0)
        assert budget.consumed_calls == 1  # No change