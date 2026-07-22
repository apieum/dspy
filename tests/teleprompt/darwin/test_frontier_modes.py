import dspy

from dspy.teleprompt.darwin import Candidate, CompositeMetric, GEPAConfig, Metric, ParetoFrontier
from dspy.teleprompt.darwin.data.cohort import Survivors


def candidate(score):
    value = Candidate(dspy.Predict("question -> answer"))
    value.scores = [score]
    return value


def test_composite_metric_keeps_objective_scores():
    metric = CompositeMetric({
        "accuracy": lambda example, prediction, trace=None: 1.0,
        "style": lambda example, prediction, trace=None: 0.5,
    })
    result = metric(dspy.Example(answer="a"), "a")

    assert result.objective_scores == {"accuracy": 1.0, "style": 0.5}
    assert float(result) == 0.75


def test_hybrid_frontier_tracks_instance_and_objectives():
    score = Metric(0.75, id="example-1", objective_scores={"accuracy": 1.0, "style": 0.5})
    selector = ParetoFrontier(GEPAConfig(frontier_type="hybrid"))
    selector.promote(Survivors(candidate(score), iteration=0))

    assert "example-1" in selector.example_best_candidates
    assert "objective:accuracy" in selector.example_best_candidates
    assert "objective:style" in selector.example_best_candidates
