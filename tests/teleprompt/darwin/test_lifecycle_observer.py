"""Tests for Darwin compilation lifecycle observers."""

import dspy

from dspy.teleprompt.darwin import Darwin, DarwinConfig, GEPAStrategy
from dspy.utils.dummies import DummyLM


class RecordingObserver:
    def __init__(self):
        self.events = []

    def start_compilation(self, student, dataset_manager):
        self.events.append(("start_compilation", dataset_manager.num_eval_tasks))

    def finish_compilation(self, result):
        self.events.append(("finish_compilation", result._compiled))

    def start_iteration(self, iteration, cohort, budget):
        self.events.append(("start_iteration", iteration))

    def finish_iteration(self, iteration, cohort, budget):
        self.events.append(("finish_iteration", iteration))


def test_strategy_notifies_compilation_observer():
    observer = RecordingObserver()
    student = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="2+2", answer="4").with_inputs("question")]

    with dspy.context(lm=DummyLM([{"answer": "4"}])):
        optimizer = Darwin(
            GEPAStrategy,
            DarwinConfig(max_lm_calls=1, observers=(observer,)),
        )
        optimizer.compile(student, trainset=trainset)

    names = [name for name, _ in observer.events]
    assert names[0] == "start_compilation"
    assert "finish_compilation" in names
