"""Test clean dependency injection design for GEPA components."""

import dspy
from dspy.teleprompt.gepa.core import GEPA
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
from dspy.teleprompt.gepa.generation.reflective_mutation_native import ReflectivePromptMutation
from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider
from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManager


def simple_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for testing."""
    if hasattr(example, 'answer') and hasattr(prediction, 'answer'):
        return 1.0 if str(example.answer) == str(prediction.answer) else 0.0
    return 0.0


def test_components_self_configure():
    """Test that components handle their own configuration."""

    # Create training data
    training_data = [
        dspy.Example(question="test1", answer="answer1").with_inputs("question"),
        dspy.Example(question="test2", answer="answer2").with_inputs("question"),
        dspy.Example(question="test3", answer="answer3").with_inputs("question"),
    ]

    # Create components without dataset knowledge
    evaluator = PromotionEvaluator(metric=simple_metric, minibatch_size=2)
    feedback_provider = FeedbackProvider(metric=simple_metric)
    generator = ReflectivePromptMutation(feedback_provider)

    # Verify components start without dataset manager
    assert evaluator.dataset_manager is None
    assert generator.dataset_manager is None

    # Components configure themselves when given dataset manager
    student = dspy.Predict("question -> answer")
    dataset_manager = DefaultDatasetManager(training_data, split_ratio=0.33)

    evaluator.start_compilation(student, dataset_manager)
    generator.start_compilation(student, dataset_manager)

    # Verify components now have dataset manager
    assert evaluator.dataset_manager is dataset_manager
    assert generator.dataset_manager is dataset_manager


def test_gepa_delegates_configuration():
    """Test that GEPA delegates configuration to components."""

    training_data = [
        dspy.Example(question="test1", answer="answer1").with_inputs("question"),
        dspy.Example(question="test2", answer="answer2").with_inputs("question"),
        dspy.Example(question="test3", answer="answer3").with_inputs("question"),
    ]

    # Create GEPA optimizer
    gepa = GEPA.create_basic(metric=simple_metric, max_calls=10)
    student = dspy.Predict("question -> answer")

    # Verify components start unconfigured
    assert gepa.evaluator.dataset_manager is None
    assert gepa.generator.dataset_manager is None

    # Mock to terminate after configuration check
    configuration_checked = False

    def mock_next_generation(parents):
        nonlocal configuration_checked
        if not configuration_checked:
            # Verify components are now configured with dataset manager
            assert gepa.evaluator.dataset_manager is not None
            assert gepa.generator.dataset_manager is not None

            # Check dataset manager has proper split
            dataset_mgr = gepa.evaluator.dataset_manager
            assert dataset_mgr.num_dev_examples >= 1
            assert dataset_mgr.num_eval_tasks >= 1
            assert dataset_mgr.num_dev_examples + dataset_mgr.num_eval_tasks == len(training_data)

            configuration_checked = True

        # Return mock result to terminate
        if not parents.is_empty():
            first_candidate = parents.first()
            first_candidate.module._compiled = True
            return first_candidate.module
        return student

    gepa.next_generation = mock_next_generation

    # Compile should trigger component configuration
    result = gepa.compile(student, training_data)

    # Verify configuration happened and result is compiled
    assert configuration_checked
    assert hasattr(result, '_compiled')


def test_dataset_split_algorithm():
    """Test paper-compliant dataset split implementation."""

    training_data = [
        dspy.Example(question=f"test{i}", answer=f"answer{i}").with_inputs("question")
        for i in range(10)
    ]

    gepa = GEPA.create_basic(metric=simple_metric, max_calls=5)
    student = dspy.Predict("question -> answer")

    # Mock to capture and verify split
    split_captured = False

    def mock_next_generation(parents):
        nonlocal split_captured
        if not split_captured:
            # Verify algorithm-compliant split through dataset manager
            dataset_mgr = gepa.evaluator.dataset_manager
            eval_size = dataset_mgr.num_eval_tasks
            dev_size = dataset_mgr.num_dev_examples

            # Should preserve total dataset size
            assert eval_size + dev_size == len(training_data)

            # Should have reasonable split (not all to one dataset)
            assert eval_size >= 1
            assert dev_size >= 1

            split_captured = True

        # Terminate quickly
        if not parents.is_empty():
            first_candidate = parents.first()
            first_candidate.module._compiled = True
            return first_candidate.module
        return student

    gepa.next_generation = mock_next_generation

    result = gepa.compile(student, training_data)

    assert split_captured
    assert hasattr(result, '_compiled')


if __name__ == "__main__":
    test_components_self_configure()
    test_gepa_delegates_configuration()
    test_dataset_split_algorithm()
    print("All dependency injection tests passed!")
