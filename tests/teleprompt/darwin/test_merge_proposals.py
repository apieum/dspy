import dspy

from dspy.teleprompt.darwin.algorithms.gepa import GEPACandidate as Candidate
from dspy.teleprompt.darwin.algorithms.gepa import GEPAConfig, SystemAwareMerge
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.utils import get_signature, set_signature
from dspy.teleprompt.darwin.evaluation import Metric


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

    merged = SystemAwareMerge(config=GEPAConfig()).generate(Parents(first, second, iteration=1))

    assert len(merged) == 1
    child = merged.first()
    assert child.parents == [first, second, ancestor]
    assert child.creation_metadata["merge_type"] == "system_aware"


def test_system_aware_merge_is_observable():
    assert hasattr(SystemAwareMerge(config=GEPAConfig()), "subscribe")


def test_system_aware_merge_does_not_reintroduce_better_ancestor():
    ancestor = Candidate(dspy.Predict("question -> answer"), generation_number=0)
    first_module = ancestor.module.deepcopy()
    second_module = ancestor.module.deepcopy()
    instruction(first_module, "First variant")
    instruction(second_module, "Second variant")
    first = Candidate(first_module, parents=[ancestor], generation_number=1)
    second = Candidate(second_module, parents=[ancestor], generation_number=1)
    ancestor.scores = [Metric(1.0, id="task")]
    first.scores = [Metric(0.5, id="task")]
    second.scores = [Metric(0.5, id="task")]

    merged = SystemAwareMerge(config=GEPAConfig()).generate(Parents(first, second, iteration=1))

    assert merged.is_empty()


def test_system_aware_merge_requires_shared_validation_support():
    ancestor = Candidate(dspy.Predict("question -> answer"), generation_number=0)
    first_module = ancestor.module.deepcopy()
    second_module = ancestor.module.deepcopy()
    instruction(first_module, "First variant")
    instruction(second_module, "Second variant")
    first = Candidate(first_module, parents=[ancestor], generation_number=1)
    second = Candidate(second_module, parents=[ancestor], generation_number=1)
    first.scores = [Metric(1.0, id="task-1")]
    second.scores = [Metric(1.0, id="task-2")]

    merged = SystemAwareMerge(
        config=GEPAConfig(merge_val_overlap_floor=1)
    ).generate(
        Parents(first, second, iteration=1)
    )

    assert merged.is_empty()


def test_system_aware_merge_preserves_shared_innovation():
    ancestor = Candidate(dspy.Predict("question -> answer"), generation_number=0)
    shared_module_1 = ancestor.module.deepcopy()
    shared_module_2 = ancestor.module.deepcopy()
    instruction(shared_module_1, "The same useful innovation.")
    instruction(shared_module_2, "The same useful innovation.")
    first = Candidate(shared_module_1, parents=[ancestor], generation_number=1)
    second = Candidate(shared_module_2, parents=[ancestor], generation_number=1)

    generator = SystemAwareMerge(config=GEPAConfig())
    desirable = generator._find_desirable_signatures(ancestor, first, second)

    assert len(desirable) == 1
    assert desirable[0][1].instructions == "The same useful innovation."


def test_system_aware_merge_selects_balanced_validation_support():
    examples = [dspy.Example(question=f"q-{idx}", answer=f"a-{idx}") for idx in range(3)]
    ancestor = Candidate(dspy.Predict("question -> answer"), generation_number=0)
    first = Candidate(ancestor.module.deepcopy(), parents=[ancestor], generation_number=1)
    second = Candidate(ancestor.module.deepcopy(), parents=[ancestor], generation_number=1)
    first.scores = [Metric(1.0, id=str(id(examples[0]))), Metric(0.0, id=str(id(examples[1]))), Metric(0.5, id=str(id(examples[2])))]
    second.scores = [Metric(0.0, id=str(id(examples[0]))), Metric(1.0, id=str(id(examples[1]))), Metric(0.5, id=str(id(examples[2])))]

    generator = SystemAwareMerge(config=GEPAConfig(), feedback_data=examples)
    generator.validation_data = [(str(id(example)), example) for example in examples]
    selected = generator._select_merge_minibatch(first, second)

    assert {example.question for example in selected} == {"q-0", "q-1", "q-2"}


def test_system_aware_merge_orders_ancestors_by_aggregate_score():
    generator = SystemAwareMerge(config=GEPAConfig())
    parent1 = Candidate(dspy.Predict("question -> answer"))
    parent2 = Candidate(dspy.Predict("question -> answer"))
    weak = Candidate(dspy.Predict("question -> answer"), generation_number=1)
    strong = Candidate(dspy.Predict("question -> answer"), generation_number=2)
    weak.scores = [Metric(0.1, id="task")]
    strong.scores = [Metric(0.8, id="task")]
    parent1.scores = [Metric(1.0, id="task")]
    parent2.scores = [Metric(1.0, id="task")]

    ordered = generator._ancestor_order({weak, strong}, parent1, parent2)

    assert set(ordered) == {weak, strong}
    assert ordered[0] is strong
