"""Test core Darwin data structures and utilities."""

import dspy
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import NewBorns, Survivors, Parents
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy


class TestCandidate:
    """Test Candidate core functionality."""

    def test_candidate_creation(self):
        """Test basic candidate creation."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)
        
        assert candidate.module == module
        assert candidate.generation_number == 1
        assert candidate.task_scores == {}

    def test_candidate_scoring(self):
        """Test candidate task scoring."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=0)
        
        candidate.task_scores = {0: 0.8, 1: 0.6, 2: 0.9}
        
        assert candidate.task_score(0) == 0.8
        assert candidate.task_score(1) == 0.6
        assert abs(candidate.average_task_score() - 0.77) < 0.01  # (0.8 + 0.6 + 0.9) / 3

    def test_candidate_domination(self):
        """Test candidate domination logic."""
        module1 = dspy.Predict("input -> output")
        module2 = dspy.Predict("input -> output")
        
        candidate_a = Candidate(module1, generation_number=0)
        candidate_b = Candidate(module2, generation_number=0)
        
        candidate_a.task_scores = {0: 0.9, 1: 0.8, 2: 0.7}  # Superior
        candidate_b.task_scores = {0: 0.3, 1: 0.2, 2: 0.1}  # Dominated
        
        assert candidate_a.dominate(candidate_b)
        assert not candidate_b.dominate(candidate_a)


class TestCohort:
    """Test Cohort functionality."""

    def test_cohort_creation(self):
        """Test cohort creation and basic operations."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=0)
        
        # Test different cohort types
        newborns = NewBorns(candidate, iteration=1)
        assert newborns.size() == 1
        assert candidate in newborns
        
        survivors = Survivors(candidate, iteration=1)
        assert survivors.size() == 1
        
        parents = Parents(candidate, iteration=1)
        assert parents.size() == 1

    def test_empty_cohort(self):
        """Test empty cohort handling."""
        empty_parents = Parents(iteration=0)
        assert empty_parents.is_empty()
        assert empty_parents.size() == 0


class TestBudget:
    """Test Budget functionality."""

    def test_budget_creation(self):
        """Test budget creation and basic operations."""
        budget = LMCallsBudget(max_calls=10)
        
        assert budget.peek() == 10
        assert budget > 5
        assert budget <= 10

    def test_budget_spending(self):
        """Test budget spending operations."""
        budget = LMCallsBudget(max_calls=10)
        initial_remaining = budget.peek()
        
        # Simulate spending on evaluation
        module = dspy.Predict("input -> output")
        budget.spend_on_evaluation(module, {"phase": "test", "examples": 2})
        
        assert budget.peek() < initial_remaining

    def test_budget_compilation_lifecycle(self):
        """Test budget lifecycle methods."""
        budget = LMCallsBudget(max_calls=10)
        student = dspy.Predict("input -> output")
        
        training_data = [dspy.Example(question="test", answer="answer").with_inputs("question")]
        split_strategy = DefaultSplitStrategy(trainset=training_data, verbose=False)
        
        # Should not crash
        budget.start_compilation(student, split_strategy=split_strategy, verbose=False)
        budget.finish_compilation(student)


class TestSplitStrategy:
    """Test SplitStrategy functionality."""

    def test_split_strategy_creation(self):
        """Test split strategy creation."""
        trainset = [
            dspy.Example(question="q1", answer="a1").with_inputs("question"),
            dspy.Example(question="q2", answer="a2").with_inputs("question"),
            dspy.Example(question="q3", answer="a3").with_inputs("question"),
        ]
        
        strategy = DefaultSplitStrategy(trainset=trainset)
        
        # Should create internal validation set (20% of 3 = 1 due to minimum)
        assert len(strategy.internal_validation_set) >= 1
        assert len(strategy.external_devset) == 0

    def test_split_strategy_with_devset(self):
        """Test split strategy with external devset."""
        trainset = [dspy.Example(question="train", answer="ans").with_inputs("question")]
        devset = [dspy.Example(question="dev", answer="dev_ans").with_inputs("question")]
        
        strategy = DefaultSplitStrategy(trainset=trainset, devset=devset)
        
        assert len(strategy.internal_validation_set) >= 1  # From trainset
        assert len(strategy.external_devset) == 1  # External devset preserved

    def test_minibatch_sampling(self):
        """Test minibatch sampling from split strategy."""
        trainset = [
            dspy.Example(question=f"q{i}", answer=f"a{i}").with_inputs("question")
            for i in range(10)
        ]
        
        strategy = DefaultSplitStrategy(trainset=trainset)
        candidate = Candidate(dspy.Predict("input -> output"), generation_number=0)
        
        # Test feedback minibatch
        feedback_batch = strategy.get_feedback_minibatch(candidate, size=2)
        assert len(feedback_batch) == 2
        assert all(example in strategy.internal_validation_set for example in feedback_batch)
        
        # Test evaluation minibatch
        eval_batch = strategy.get_evaluation_minibatch(candidate, size=1)
        assert len(eval_batch) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])