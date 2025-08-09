"""Test GEPA dataset split implementation."""

import dspy
from dspy.teleprompt.gepa.core import GEPA
from dspy.teleprompt.gepa.generation.reflective_mutation_native import ReflectivePromptMutation
from dspy.teleprompt.gepa.generation.feedback import FeedbackProvider
from dspy.teleprompt.gepa.evaluation.promotion import PromotionEvaluator
from dspy.teleprompt.gepa.dataset_manager import DefaultDatasetManagerFactory, DefaultDatasetManager


class TestDatasetSplit:
    """Test GEPA dataset splitting."""

    def test_compile_splits_dataset_correctly(self):
        """Test that compile() splits dataset correctly."""

        # Create test dataset
        training_data = [
            dspy.Example(question=f'example_{i}', answer=f'answer_{i}').with_inputs('question')
            for i in range(10)
        ]

        # Create simple student
        student = dspy.Predict("question -> answer")

        # Create GEPA with components that track dataset usage
        def simple_metric(example, prediction, trace=None):
            return 0.5

        feedback_provider = FeedbackProvider(metric=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider)
        evaluator = PromotionEvaluator(metric=simple_metric)

        from dspy.teleprompt.gepa.budget.llm_calls import LLMCallsBudget
        from dspy.teleprompt.gepa.selection import ParetoFrontier

        gepa = GEPA(
            budget=LLMCallsBudget(5),  # Small budget to terminate quickly
            selector=ParetoFrontier(),
            generator=generator,
            evaluator=evaluator,
            dataset_manager_factory=DefaultDatasetManagerFactory()
        )

        # Mock optimization to terminate quickly and verify dataset manager is used
        compilation_started = False

        original_start_compilation = gepa.start_compilation
        def mock_start_compilation(student, dataset_manager):
            nonlocal compilation_started
            compilation_started = True

            # Verify dataset manager has the right split
            pareto_set = dataset_manager.get_eval_set()
            feedback_batch = dataset_manager.get_feedback_minibatch(5)

            # Should have data for both components
            assert len(pareto_set) >= 1
            assert len(feedback_batch) >= 1

            # Total examples should not exceed original (sampling may reduce it)
            assert len(pareto_set) <= len(training_data)

            return original_start_compilation(student, dataset_manager)

        gepa.start_compilation = mock_start_compilation

        # Mock next_generation to terminate immediately
        original_next_gen = gepa.next_generation
        def mock_next_gen(parents):
            # Return mock result to terminate
            if not parents.is_empty():
                mock_candidate = parents.first()
                mock_candidate.module._compiled = True
                return mock_candidate.module
            return student

        gepa.next_generation = mock_next_gen

        # Compile should trigger split
        result = gepa.compile(student, training_data)

        # Verify result and that compilation started (dataset manager was used)
        assert compilation_started, "Dataset manager should have been created and used"
        assert hasattr(result, '_compiled')

    def test_generator_uses_dev_data(self):
        """Test that Generator receives and uses dataset manager."""

        feedback_provider = FeedbackProvider(metric=lambda ex, pred, trace=None: 0.5)
        generator = ReflectivePromptMutation(feedback_provider)

        # Create training data that will be split by dataset manager
        training_data = [
            dspy.Example(input=f'example_{i}', answer=f'answer_{i}').with_inputs('input')
            for i in range(10)
        ]

        # Create dataset manager
        dataset_manager = DefaultDatasetManager(training_data)

        # Call start_compilation with dataset manager
        student = dspy.Predict("input -> answer")
        generator.start_compilation(student, dataset_manager)

        # Verify generator has access to dataset manager
        assert generator.dataset_manager is dataset_manager

        # Verify it can get feedback minibatches
        feedback_batch = generator.dataset_manager.get_feedback_minibatch(3)
        assert len(feedback_batch) <= 3
        assert len(feedback_batch) >= 1  # Should have at least some data

    def test_evaluator_uses_both_datasets(self):
        """Test that Evaluator receives and uses dataset manager."""

        evaluator = PromotionEvaluator(metric=lambda ex, pred, trace=None: 0.5)

        # Create training data that will be split by dataset manager
        training_data = [
            dspy.Example(input=f'example_{i}', answer=f'answer_{i}').with_inputs('input')
            for i in range(10)
        ]

        # Create dataset manager
        dataset_manager = DefaultDatasetManager(training_data)

        # Call start_compilation with dataset manager
        student = dspy.Predict("input -> answer")
        evaluator.start_compilation(student, dataset_manager)

        # Verify evaluator has access to dataset manager
        assert evaluator.dataset_manager is dataset_manager

        # Verify it can get both pareto set and validation minibatches
        pareto_set = evaluator.dataset_manager.get_eval_set()
        validation_batch = evaluator.dataset_manager.get_validation_minibatch(3)

        assert len(pareto_set) >= 1  # Should have pareto data
        assert len(validation_batch) <= 3  # Should respect minibatch size
        assert len(validation_batch) >= 1  # Should have some validation data

    def test_minimum_dataset_sizes(self):
        """Test that dataset split ensures minimum sizes."""

        # Very small dataset
        training_data = [
            dspy.Example(question='q1', answer='a1').with_inputs('question'),
            dspy.Example(question='q2', answer='a2').with_inputs('question'),
        ]

        student = dspy.Predict("question -> answer")
        gepa = GEPA.create_basic(lambda ex, pred, trace=None: 0.5, max_calls=2)

        # Mock to check dataset manager handles small datasets properly
        original_start = gepa.start_compilation
        def mock_start(student, dataset_manager):
            # Should have at least 1 example in both splits
            pareto_set = dataset_manager.get_eval_set()
            feedback_batch = dataset_manager.get_feedback_minibatch(5)

            assert len(pareto_set) >= 1, "Should have at least 1 pareto example"
            assert len(feedback_batch) >= 1, "Should have at least 1 feedback example"

            # For small datasets, DefaultDatasetManager should handle gracefully
            # Total unique examples may be equal to training data for small sets
            return original_start(student, dataset_manager)

        gepa.start_compilation = mock_start

        # Should not fail
        result = None
        try:
            result = gepa.compile(student, training_data)
            assert hasattr(result, '_compiled')
        except Exception as e:
            # Budget exhaustion is expected, but split should work
            assert 'Budget exhausted' in str(e) or hasattr(result, '_compiled')
