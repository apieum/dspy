"""Proof that Darwin phase contracts do not require GEPA cohort roles."""

from dspy.teleprompt.darwin.data import Candidate, Cohort
from dspy.teleprompt.darwin.generation import Generator
from dspy.teleprompt.darwin.evaluation import Evaluator
from dspy.teleprompt.darwin.selection import Selector


class Generated(Cohort):
    pass


class Evaluated(Cohort):
    pass


class IntGenerator(Generator):
    def generate(self, parents, budget=None):
        return Generated(Candidate(parents.first().value + 1), iteration=parents.iteration)


class FirstEvaluator(Evaluator):
    def __init__(self, *, config, **kwargs):
        super().__init__()

    def evaluate(self, cohort, budget):
        return Generated(*cohort, iteration=cohort.iteration)


class SecondEvaluator(Evaluator):
    def __init__(self, *, config, **kwargs):
        super().__init__()

    def evaluate(self, cohort, budget):
        return Evaluated(*cohort, iteration=cohort.iteration)


class IntSelector(Selector):
    def __init__(self):
        super().__init__()
        self.population = []

    def size(self):
        return len(self.population)

    def promote(self, cohort, budget=None):
        self.population = cohort.to_list()
        return Evaluated(*self.population, iteration=cohort.iteration)

    def best_candidate(self):
        return max(self.population, key=lambda candidate: candidate.value)


def test_generator_batch_uses_the_concrete_output_cohort():
    parents = Cohort(Candidate(1), iteration=3)

    generated = IntGenerator().generate_batch(parents, 2)

    assert isinstance(generated, Generated)
    assert [candidate.value for candidate in generated] == [2, 2]


def test_evaluator_chain_passes_cohorts_without_role_conversion():
    initial = Cohort(Candidate(1), iteration=4)
    chain_type = Evaluator.create_chain("TwoStages", [FirstEvaluator, SecondEvaluator])

    evaluated = chain_type(config=object()).evaluate(initial, budget=object())

    assert isinstance(evaluated, Evaluated)
    assert [candidate.value for candidate in evaluated] == [1]


def test_selector_size_is_defined_by_the_concrete_algorithm():
    selector = IntSelector()
    cohort = Evaluated(Candidate(4), iteration=0)

    selected = selector.promote(cohort)

    assert selector.size() == 1
    assert selector.best_candidate().value == 4
    assert isinstance(selected, Evaluated)
