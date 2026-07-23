import dspy

from dspy.teleprompt.darwin import (
    AnyStopper,
    FileStopper,
    ScoreThresholdStopper,
)
from dspy.teleprompt.darwin.algorithms.gepa import GEPAConfig
from dspy.teleprompt.darwin.algorithms.gepa import GEPACandidate as Candidate
from dspy.teleprompt.darwin.evaluation.metrics import Metric
from dspy.teleprompt.darwin.algorithms.gepa import GEPAStrategy


def test_score_threshold_stopper_uses_best_candidate():
    strategy = GEPAStrategy(GEPAConfig(
        max_lm_calls=1,
        stoppers=(ScoreThresholdStopper(1.0),),
    ))
    strategy.best_candidate = Candidate(dspy.Predict("question -> answer"))
    strategy.best_candidate.scores = [Metric(1.0, id="task")]

    assert ScoreThresholdStopper(1.0)(strategy)
    assert strategy.should_terminate()


def test_file_and_composite_stoppers(tmp_path):
    strategy = GEPAStrategy(GEPAConfig(max_lm_calls=1))
    control_file = tmp_path / "stop"
    stopper = AnyStopper(FileStopper(control_file))
    assert not stopper(strategy)
    control_file.touch()
    assert stopper(strategy)
