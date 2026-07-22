import dspy
import pytest

from dspy.teleprompt.darwin import FullEvaluationPolicy, MinibatchEvaluationPolicy


def examples(count):
    return [dspy.Example(value=i).with_inputs("value") for i in range(count)]


def test_full_validation_policy_returns_all_examples_in_order():
    data = examples(4)
    assert FullEvaluationPolicy().get_eval_batch(data) == data


def test_minibatch_policy_is_seeded_per_iteration():
    data = examples(10)
    policy = MinibatchEvaluationPolicy(batch_size=3, seed=4)
    assert policy.get_eval_batch(data, iteration=2) == policy.get_eval_batch(data, iteration=2)
    assert len(policy.get_eval_batch(data, iteration=2)) == 3


def test_minibatch_policy_validates_batch_size():
    with pytest.raises(ValueError):
        MinibatchEvaluationPolicy(batch_size=0)
