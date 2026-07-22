import dspy

from dspy.teleprompt.darwin import (
    ConfidenceAssessor,
    LinearConfidenceScoring,
    Metric,
    ThresholdConfidenceScoring,
)


def test_confidence_scoring_degrades_to_original_without_logprob():
    assessor = ConfidenceAssessor(lambda e, p, t=None: Metric(1.0))
    result = assessor(dspy.Example(answer="a"), "a")

    assert result.value == 1.0
    assert result.side_info["logprob"] is None


def test_confidence_scoring_penalizes_uncertain_correct_answer():
    assessor = ConfidenceAssessor(
        lambda e, p, t=None: Metric(1.0, side_info={"joint_logprob": -2.0}),
        scoring=LinearConfidenceScoring(threshold=0.5, minimum=0.3),
    )
    result = assessor(dspy.Example(answer="a"), "a")

    assert 0.3 < result.value < 1.0
    assert result.side_info["confidence_adjusted"] is True


def test_threshold_confidence_keeps_incorrect_answers_at_zero():
    scoring = ThresholdConfidenceScoring(threshold=0.5)
    assert scoring.score(0.0, 0.0) == 0.0
