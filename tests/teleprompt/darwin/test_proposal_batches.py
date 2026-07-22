import dspy

from dspy.teleprompt.darwin import Candidate, Generator
from dspy.teleprompt.darwin.data.cohort import NewBorns, Parents


class CountingGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, parents, budget=None):
        self.calls += 1
        return NewBorns(
            Candidate(dspy.Predict("question -> answer"), parents=list(parents)),
            iteration=parents.iteration,
        )


def test_generator_batch_is_a_first_class_proposal_interface():
    parent = Candidate(dspy.Predict("question -> answer"))
    parents = Parents(parent, iteration=2)
    generator = CountingGenerator()

    proposals = generator.generate_batch(parents, 3)

    assert generator.calls == 3
    assert len(proposals) == 3
    assert all(candidate.parents == [parent] for candidate in proposals)
    assert all(candidate.generation_number == 0 for candidate in proposals)
