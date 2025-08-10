"""Test compilation lifecycle observer pattern."""

import pytest
import dspy
from dspy.teleprompt.darwin.optimizer import Darwin, GEPAMute
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.dataset_manager import DefaultDatasetManager
from unittest.mock import Mock


# @pytest.mark.slow_test
def test_compilation_lifecycle_events():
    """Test that components receive compilation lifecycle events."""

    def simple_metric(example, prediction, trace=None):
        return 0.5

    # Create GEPA with lifecycle tracking
    darwin = GEPAMute(simple_metric, max_calls=5)

    # Track lifecycle calls
    lifecycle_calls = []

    # Mock components to track lifecycle
    for component in [darwin.budget, darwin.selector, darwin.generator, darwin.evaluator]:
        original_start = component.start_compilation
        original_finish = component.finish_compilation

        def make_tracked_start(comp_name, orig_start):
            def tracked_start(*args, **kwargs):
                lifecycle_calls.append(f"{comp_name}_start_compilation")
                return orig_start(*args, **kwargs)
            return tracked_start

        def make_tracked_finish(comp_name, orig_finish):
            def tracked_finish(*args, **kwargs):
                lifecycle_calls.append(f"{comp_name}_finish_compilation")
                return orig_finish(*args, **kwargs)
            return tracked_finish

        component.start_compilation = make_tracked_start(component.__class__.__name__, original_start)
        component.finish_compilation = make_tracked_finish(component.__class__.__name__, original_finish)

    # Create test data
    training_data = [
        dspy.Example(question="test", answer="answer").with_inputs("question")
    ]
    student = dspy.Predict("question -> answer")

    # Run compilation (should trigger lifecycle events)
    with dspy.context(lm=dspy.utils.DummyLM({"answer": "test"})):
        result = darwin.compile(student, training_data)

    # Verify lifecycle events occurred
    assert any("start_compilation" in call for call in lifecycle_calls)
    assert any("finish_compilation" in call for call in lifecycle_calls)
    assert hasattr(result, '_compiled')


def test_budget_tracks_compilation_progress():
    """Test that budget tracks compilation progress correctly."""

    def simple_metric(example, prediction, trace=None):
        return 0.5

    budget = LMCallsBudget(20)

    # Create test data
    training_data = [
        dspy.Example(question="test", answer="answer").with_inputs("question"),
        dspy.Example(question="test2", answer="answer2").with_inputs("question"),
    ]
    student = dspy.Predict("question -> answer")

    # Test start compilation with DatasetManager
    dataset_manager = DefaultDatasetManager(training_data)
    budget.start_compilation(student, dataset_manager)
    assert budget.consumed_calls == 0

    # Test spending budget - simulate module with history
    initial_calls = budget.consumed_calls
    # Simulate a module that has made some LLM calls
    student.history = [{"call": 1}, {"call": 2}]  # Mock history with 2 calls
    budget.spend_on_evaluation(student, {"phase": "test", "examples": 2})
    assert budget.consumed_calls > initial_calls

    # Test finish compilation
    budget.finish_compilation(student)
    # Should not crash and should have tracked usage


def test_components_opt_into_lifecycle_events():
    """Test that components can implement lifecycle events."""

    def simple_metric(example, prediction, trace=None):
        return 0.5

    # Create components
    from dspy.teleprompt.darwin.generation.mutation import ReflectivePromptMutation
    from dspy.teleprompt.darwin.generation.feedback import FeedbackProvider
    from dspy.teleprompt.darwin.evaluation.gepa_evaluator import GEPATwoPhasesEval
    from dspy.teleprompt.darwin.selection.pareto import ParetoFrontier

    feedback_provider = FeedbackProvider(metric=simple_metric)
    generator = ReflectivePromptMutation(feedback_provider)
    evaluator = GEPATwoPhasesEval(metric=simple_metric)
    selector = ParetoFrontier()
    budget = LMCallsBudget(10)

    # All components should have lifecycle methods
    components = [generator, evaluator, selector, budget]

    for component in components:
        assert hasattr(component, 'start_compilation')
        assert hasattr(component, 'finish_compilation')
        assert hasattr(component, 'start_iteration')
        assert hasattr(component, 'finish_iteration')

    # Test that lifecycle methods can be called
    student = dspy.Predict("question -> answer")
    training_data = [dspy.Example(question="test", answer="answer").with_inputs("question")]

    # Create DatasetManager for lifecycle method calls
    dataset_manager = DefaultDatasetManager(training_data)

    # Should not crash
    generator.start_compilation(student, dataset_manager)
    evaluator.start_compilation(student, dataset_manager)
    selector.start_compilation(student, dataset_manager)
    budget.start_compilation(student, dataset_manager)

    # Lifecycle methods should complete successfully
    for component in components:
        component.start_iteration(0, Mock(), budget)
        component.finish_iteration(0, Mock(), budget)
        component.finish_compilation(student)


if __name__ == "__main__":
    test_compilation_lifecycle_events()
    test_budget_tracks_compilation_progress()
    test_components_opt_into_lifecycle_events()
    print("All lifecycle observer tests passed!")
