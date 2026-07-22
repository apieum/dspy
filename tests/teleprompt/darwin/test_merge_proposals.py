import dspy

from dspy.teleprompt.darwin import Candidate, SystemAwareMerge
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.utils import get_signature, set_signature


def instruction(module, value):
    predictor = module.predictors()[0]
    signature = get_signature(predictor).with_instructions(value)
    set_signature(predictor, signature)


def test_system_aware_merge_combines_divergent_lineages():
    ancestor = Candidate(dspy.Predict("question -> answer"), generation_number=0)
    first_module = ancestor.module.deepcopy()
    second_module = ancestor.module.deepcopy()
    instruction(first_module, "Answer with careful reasoning.")
    instruction(second_module, "Answer with concise facts.")
    first = Candidate(first_module, parents=[ancestor], generation_number=1)
    second = Candidate(second_module, parents=[ancestor], generation_number=1)
    first.scores = []
    second.scores = []

    merged = SystemAwareMerge().generate(Parents(first, second, iteration=1))

    assert len(merged) == 1
    child = merged.first()
    assert child.parents == [first, second, ancestor]
    assert child.creation_metadata["merge_type"] == "system_aware"


def test_system_aware_merge_is_observable():
    assert hasattr(SystemAwareMerge(), "subscribe")
