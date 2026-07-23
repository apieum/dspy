"""Tests for Darwin's centralized dataset management."""

import dspy
from dspy.teleprompt.darwin.algorithms.gepa import GEPAConfig
from dspy.teleprompt.darwin.evaluation.cache import EvaluationCache

from dspy.teleprompt.darwin.dataset_manager import (
    DefaultDatasetManager,
    DefaultDatasetManagerFactory,
)


def _examples(count):
    return [
        dspy.Example(question=f"q-{index}", answer=f"a-{index}").with_inputs("question")
        for index in range(count)
    ]


def test_explicit_validation_data_is_kept_separate():
    trainset = _examples(4)
    devset = _examples(2)
    manager = DefaultDatasetManager(trainset, devset)

    assert len(manager.get_eval_set()) == 4
    assert manager.num_dev_examples == 2
    assert len(manager.get_validation_minibatch(10)) == 2


def test_small_dataset_is_safe_without_explicit_validation_data():
    manager = DefaultDatasetManager(_examples(1))

    assert manager.num_eval_tasks == 1
    assert manager.num_dev_examples == 1
    assert len(manager.get_feedback_minibatch(1)) == 1


def test_factory_preserves_configured_split_ratio():
    manager = DefaultDatasetManagerFactory(split_ratio=0.5).create(_examples(10))

    assert manager.num_eval_tasks == 5
    assert manager.num_dev_examples == 5


def test_seeded_feedback_sampling_is_reproducible():
    examples = _examples(10)
    first = DefaultDatasetManager(examples, seed=7).get_feedback_minibatch(3)
    second = DefaultDatasetManager(examples, seed=7).get_feedback_minibatch(3)

    assert list(first) == list(second)


def test_seeded_automatic_split_is_reproducible_and_shuffled():
    examples = _examples(10)
    first = DefaultDatasetManager(examples, split_ratio=0.5, seed=7)
    second = DefaultDatasetManager(examples, split_ratio=0.5, seed=7)

    assert list(first.get_eval_set()) == list(second.get_eval_set())
    assert list(first.get_eval_set()) != [0, 1, 2, 3, 4]


def test_components_receive_the_dataset_manager():
    from dspy.teleprompt.darwin.algorithms.gepa.evaluation import GEPATwoPhasesEval
    from dspy.teleprompt.darwin.algorithms.gepa.generation import ReflectivePromptMutation
    from dspy.teleprompt.darwin.algorithms.gepa.generation.feedback import FeedbackProvider

    manager = DefaultDatasetManager(_examples(6), seed=3)
    student = dspy.Predict("question -> answer")
    assessor = lambda example, prediction, trace=None: 0.5

    generator = ReflectivePromptMutation(
        config=GEPAConfig(),
        feedback_provider=FeedbackProvider(assessor=assessor)
    )
    generator.start_compilation(student, manager)

    evaluator = GEPATwoPhasesEval(
        config=GEPAConfig(fitness_function=assessor),
        evaluation_cache=EvaluationCache(),
        assessor=assessor,
    )
    evaluator.start_compilation(student, manager)

    assert generator.dataset_manager is manager
    assert len(generator.feedback_data) <= generator.minibatch_size
    assert evaluator.dataset_manager is manager


def test_strategy_exposes_full_training_pool_for_reflection_sampling():
    """GEPA must sample fresh reflection batches from all training examples."""
    from dspy.teleprompt.darwin.algorithms.gepa import GEPAConfig, GEPAStrategy

    student = dspy.Predict("question -> answer")
    examples = _examples(8)
    strategy = GEPAStrategy(GEPAConfig(minibatch_size=2, max_lm_calls=1))
    strategy.start_compilation(student, trainset=examples, devset=_examples(2))

    generator = strategy.generator
    assert len(generator.feedback_data) == 2
    assert generator.feedback_pool == strategy.training_data
    assert len(generator.feedback_pool) == len(strategy.training_data)


def test_strategy_accepts_preconfigured_factory_instance():
    """A factory instance should not be called as if it were a class."""
    from dspy.teleprompt.darwin import Darwin
    from dspy.teleprompt.darwin.algorithms.gepa import GEPAConfig, GEPAStrategy
    from dspy.utils.dummies import DummyLM

    student = dspy.Predict("question -> answer")
    factory = DefaultDatasetManagerFactory(split_ratio=0.5)

    with dspy.context(lm=DummyLM([{"answer": "4"}])):
        optimizer = Darwin(
            GEPAStrategy,
            GEPAConfig(max_lm_calls=1, dataset_manager_factory=factory),
        )
        optimizer.compile(student, trainset=_examples(4))
