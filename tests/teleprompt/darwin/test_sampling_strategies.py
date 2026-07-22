import random

import dspy

from dspy.teleprompt.darwin import Candidate
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.darwin.generation import (
    EpochShuffledBatchSampler,
    IndependentSampling,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
)


def _parents():
    return Parents(
        Candidate(dspy.Predict("question -> answer")),
        Candidate(dspy.Predict("question -> answer")),
        iteration=3,
        task_wins=lambda candidate: 1.0,
    )


def _ids(tasks):
    return [next(iter(task)).module for task in tasks]


def test_single_mutation_is_one_task_by_default():
    parents = _parents()
    tasks = SingleMutationSampling().sample(parents, rng=random.Random(1))
    assert len(tasks) == 1
    assert tasks[0] is parents


def test_same_parent_repeats_one_selected_parent():
    tasks = SameParentSampling(3).sample(_parents(), rng=random.Random(1))
    assert len(tasks) == 3
    assert len({id(next(iter(task))) for task in tasks}) == 1


def test_independent_sampling_creates_n_tasks():
    tasks = IndependentSampling(4).sample(_parents(), rng=random.Random(1))
    assert len(tasks) == 4
    assert all(len(task) == 1 for task in tasks)


def test_pxn_sampling_creates_p_times_n_tasks():
    tasks = PxNSampling(p=2, n=3).sample(_parents(), rng=random.Random(1))
    assert len(tasks) == 6
    assert len({id(next(iter(task))) for task in tasks[:3]}) == 1
    assert len({id(next(iter(task))) for task in tasks[3:]}) == 1


def test_sampling_rejects_non_positive_counts():
    for strategy in (SameParentSampling, IndependentSampling):
        try:
            strategy(0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_epoch_batch_sampler_returns_distinct_chunks_before_wrapping():
    sampler = EpochShuffledBatchSampler(minibatch_size=2)
    data = list(range(6))
    batches = sampler.sample(data, 3, rng=random.Random(4))

    assert sorted(value for batch in batches for value in batch) == data
    assert all(len(batch) == 2 for batch in batches)


def test_epoch_batch_sampler_is_reproducible():
    data = list(range(8))
    first = EpochShuffledBatchSampler(2).sample(data, 4, rng=random.Random(9))
    second = EpochShuffledBatchSampler(2).sample(data, 4, rng=random.Random(9))
    assert first == second
