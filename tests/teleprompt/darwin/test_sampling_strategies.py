import random

import dspy

from dspy.teleprompt.darwin.algorithms.gepa import GEPACandidate as Candidate
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.darwin.generation import (
    EpochShuffledBatchSampler,
    Generator,
    IndependentSampling,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
)
from dspy.teleprompt.darwin.data.cohort import NewBorns


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


def test_epoch_batch_sampler_pads_small_datasets_to_full_batches():
    sampler = EpochShuffledBatchSampler(minibatch_size=3)

    batches = sampler.sample(["only"], 2, rng=random.Random(3))

    assert all(len(batch) == 3 for batch in batches)
    assert all(set(batch) == {"only"} for batch in batches)


class TaskRecordingGenerator(Generator):
    def __init__(self, feedback_data):
        super().__init__()
        self.feedback_data = feedback_data
        self.tasks = []

    def generate(self, parents, budget=None):
        self.tasks.append(self._active_proposal_task)
        return NewBorns(
            Candidate(dspy.Predict("question -> answer"), parents=list(parents)),
            iteration=parents.iteration,
        )


def test_generate_batch_exposes_explicit_parent_and_minibatch_tasks():
    generator = TaskRecordingGenerator(list(range(6)))
    generator.generate_batch(
        _parents(),
        2,
        sampling_strategy=SameParentSampling(2),
        batch_sampler=EpochShuffledBatchSampler(2),
        rng=random.Random(5),
    )

    assert len(generator.tasks) == 2
    assert all(task.parents.size() == 1 for task in generator.tasks)
    assert all(len(task.feedback_data) == 2 for task in generator.tasks)
    assert generator.tasks[0].feedback_data != generator.tasks[1].feedback_data
