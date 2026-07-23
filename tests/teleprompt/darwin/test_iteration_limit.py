import dspy

from dspy.teleprompt.darwin import Darwin
from dspy.teleprompt.darwin.algorithms.gepa import GEPAConfig, GEPAStrategy
from dspy.utils.dummies import DummyLM


def test_max_iterations_is_enforced():
    data = [dspy.Example(question="q", answer="a").with_inputs("question")]
    with dspy.context(lm=DummyLM([{"answer": "a"}, {"response": "improve"}] * 4)):
        optimizer = Darwin(
            GEPAStrategy,
            GEPAConfig(max_lm_calls=50, max_iterations=0),
        )
        optimizer.compile(dspy.Predict("question -> answer"), trainset=data)

    assert optimizer.strategy.current_generation == 0
