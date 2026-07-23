import dspy
import pytest

from dspy.teleprompt.darwin import Darwin, GEPAConfig, GEPAStrategy
from dspy.utils.dummies import DummyLM


def test_darwin_accepts_standard_teleprompter_valset_keyword():
    trainset = [dspy.Example(question="train", answer="a").with_inputs("question")]
    valset = [dspy.Example(question="validation", answer="b").with_inputs("question")]
    with dspy.context(lm=DummyLM([{"answer": "a"}, {"answer": "b"}])):
        optimizer = Darwin(GEPAStrategy, GEPAConfig(max_lm_calls=2))
        optimizer.compile(dspy.Predict("question -> answer"), trainset=trainset, valset=valset)

    assert optimizer.strategy.workflow.devset == valset


def test_darwin_rejects_ambiguous_validation_aliases():
    trainset = [dspy.Example(question="train", answer="a").with_inputs("question")]
    valset = [dspy.Example(question="validation", answer="b").with_inputs("question")]
    with pytest.raises(Exception, match="either valset or devset"):
        Darwin(GEPAStrategy, GEPAConfig()).compile(
            dspy.Predict("question -> answer"),
            trainset=trainset,
            valset=valset,
            devset=valset,
        )
