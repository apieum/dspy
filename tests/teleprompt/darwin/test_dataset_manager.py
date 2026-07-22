"""Tests for Darwin's centralized dataset management."""

import dspy

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


def test_strategy_accepts_preconfigured_factory_instance():
    """A factory instance should not be called as if it were a class."""
    from dspy.teleprompt.darwin import Darwin, DarwinConfig, GEPAStrategy
    from dspy.utils.dummies import DummyLM

    student = dspy.Predict("question -> answer")
    factory = DefaultDatasetManagerFactory(split_ratio=0.5)

    with dspy.context(lm=DummyLM([{"answer": "4"}])):
        optimizer = Darwin(
            GEPAStrategy,
            DarwinConfig(max_lm_calls=1, dataset_manager_factory=factory),
        )
        optimizer.compile(student, trainset=_examples(4))
