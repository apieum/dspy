import dspy

from dspy.teleprompt.darwin.algorithms.gepa import GEPACandidate as Candidate
from dspy.teleprompt.darwin.algorithms.gepa.archive import DiversityArchive
from dspy.teleprompt.utils import get_signature, set_signature


def make_candidate(instruction, score):
    module = dspy.Predict("question -> answer")
    predictor = module.predictors()[0]
    set_signature(predictor, get_signature(predictor).with_instructions(instruction))
    candidate = Candidate(module)
    candidate.scores = [type("Score", (), {"value": score})()]
    return candidate


def test_archive_keeps_best_candidate_per_prompt_fingerprint():
    archive = DiversityArchive(capacity=3)
    weaker = make_candidate("same", 0.2)
    stronger = make_candidate("same", 0.8)
    archive.add(weaker)
    archive.add(stronger)

    assert len(archive) == 1
    assert archive.candidates()[0] is stronger


def test_archive_preserves_distinct_candidates_up_to_capacity():
    archive = DiversityArchive(capacity=2)
    candidates = [make_candidate(f"prompt-{idx}", idx / 10) for idx in range(3)]
    archive.add_all(candidates)

    assert len(archive) == 2
    assert {candidate.average_score() for candidate in archive.candidates()} == {0.1, 0.2}
