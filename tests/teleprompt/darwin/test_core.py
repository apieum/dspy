"""Test core Darwin data structures and utilities."""

import dspy
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.data.cohort import NewBorns, Survivors, Parents
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy
from dspy.teleprompt.darwin.evaluation.metrics import Metric


class TestCandidate:
    """Test Candidate core functionality."""

    def test_candidate_creation(self):
        """Test basic candidate creation."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=1)

        assert candidate.module == module
        assert candidate.generation_number == 1
        assert candidate.scores == []

    def test_candidate_scoring(self):
        """Test candidate fitness scoring with new Metric system."""
        module = dspy.Predict("input -> output")
        candidate = Candidate(module, generation_number=0)

        # Create Metric objects with UUID traces
        metric1 = Metric(0.8, id='uuid-1', trace={'dspy_uuid': 'uuid-1'})
        metric2 = Metric(0.6, id='uuid-2', trace={'dspy_uuid': 'uuid-2'})
        metric3 = Metric(0.9, id='uuid-3', trace={'dspy_uuid': 'uuid-3'})
        candidate.scores = [metric1, metric2, metric3]

        assert candidate.find_score_by_uuid('uuid-1').value == 0.8
        assert candidate.find_score_by_uuid('uuid-2').value == 0.6
        assert abs(candidate.average_score() - 0.77) < 0.01  # (0.8 + 0.6 + 0.9) / 3
        assert candidate.total_score() == 2.3

    def test_candidate_domination(self):
        """Test candidate domination logic with UUID-based metrics."""
        module1 = dspy.Predict("input -> output")
        module2 = dspy.Predict("input -> output")

        candidate_a = Candidate(module1, generation_number=0)
        candidate_b = Candidate(module2, generation_number=0)

        # Create Metric objects for candidate A (superior)
        candidate_a.scores = [
            Metric(0.9, id='uuid-1'),
            Metric(0.8, id='uuid-2'),
            Metric(0.7, id='uuid-3')
        ]

        # Create Metric objects for candidate B (dominated)
        candidate_b.scores = [
            Metric(0.3, id='uuid-1'),
            Metric(0.2, id='uuid-2'),
            Metric(0.1, id='uuid-3')
        ]

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

        assert budget == 10
        assert budget > 5
        assert budget <= 10

    def test_budget_spending(self):
        """Test budget spending operations."""
        budget = LMCallsBudget(max_calls=10)
        initial_remaining = int(budget)

        # Simulate spending on evaluation
        module = dspy.Predict("input -> output")
        budget.spend_on_evaluation(module, {"phase": "test", "examples": 2})

        assert budget < initial_remaining

    def test_budget_basic_functionality(self):
        """Test budget basic functionality."""
        budget = LMCallsBudget(max_calls=10)

        # Test initial state
        assert budget > 0
        assert budget.max_calls == 10
        assert budget.consumed_calls == 0

        # Test spending budget
        module = dspy.Predict("input -> output")
        budget.spend_on_evaluation(module, {"phase": "validation", "cost": 5})
        assert budget.consumed_calls == 5
        assert budget > 0

        # Test exhaustion
        budget.spend_on_evaluation(module, {"phase": "validation", "cost": 6})
        assert budget <= 0


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


def test_total_score_is_used_for_partial_candidate_comparisons():
    first = Candidate(dspy.Predict("question -> answer"))
    second = Candidate(dspy.Predict("question -> answer"))
    first.scores = [Metric(0.8, id="a"), Metric(0.8, id="b")]
    second.scores = [Metric(1.0, id="a")]

    assert first.total_score() > second.total_score()
    assert first.best_overall(second) is first


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
