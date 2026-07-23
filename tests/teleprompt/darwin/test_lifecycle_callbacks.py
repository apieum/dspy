import dspy

from dspy.teleprompt.darwin import Darwin, GEPAConfig, GEPAStrategy
from dspy.utils.dummies import DummyLM


class RecordingObserver:
    def __init__(self):
        self.events = []

    def start_compilation(self, student, dataset_manager):
        self.events.append("start_compilation")

    def finish_compilation(self, result):
        self.events.append("finish_compilation")

    def start_iteration(self, iteration, cohort, budget):
        self.events.append("start_iteration")

    def finish_iteration(self, iteration, cohort, budget):
        self.events.append("finish_iteration")

    def candidate_evaluated(self, candidate, accepted):
        self.events.append(("candidate_evaluated", accepted))

    def budget_exhausted(self, budget):
        self.events.append("budget_exhausted")

    def next_step(self, strategy, continuing):
        self.events.append(("next_step", continuing))


def test_compilation_observer_receives_candidate_lifecycle_events():
    observer = RecordingObserver()
    student = dspy.Predict("question -> answer")
    data = [dspy.Example(question="q", answer="a").with_inputs("question")]

    with dspy.context(lm=DummyLM([{"answer": "a"}])):
        Darwin(
            GEPAStrategy,
            GEPAConfig(max_lm_calls=1, observers=(observer,)),
        ).compile(student, trainset=data)

    assert observer.events[0] == "start_compilation"
    assert any(event[0] == "candidate_evaluated" for event in observer.events if isinstance(event, tuple))
    assert any(event[0] == "next_step" for event in observer.events if isinstance(event, tuple))
    assert observer.events[-1] == "finish_compilation"
