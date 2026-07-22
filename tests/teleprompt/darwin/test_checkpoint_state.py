import json

import dspy

from dspy.teleprompt.darwin import Darwin, DarwinConfig, GEPAStrategy, OptimizationCheckpoint
from dspy.utils.dummies import DummyLM


def test_checkpoint_manifest_is_written_and_json_safe(tmp_path):
    checkpoint_path = tmp_path / "darwin-checkpoint.json"
    student = dspy.Predict("question -> answer")
    data = [dspy.Example(question="q", answer="a").with_inputs("question")]

    with dspy.context(lm=DummyLM([{"answer": "a"}])):
        Darwin(
            GEPAStrategy,
            DarwinConfig(max_lm_calls=1, checkpoint_path=str(checkpoint_path)),
        ).compile(student, trainset=data)

    payload = json.loads(checkpoint_path.read_text())
    checkpoint = OptimizationCheckpoint.from_dict(payload)
    assert checkpoint.schema_version == 1
    assert checkpoint.completed is True
    assert checkpoint.history
    assert "calls" in checkpoint.budget
    assert checkpoint.candidates
    assert "instructions" in checkpoint.candidates[0]
    assert "cohort_attributes" in checkpoint.candidates[0]
    assert checkpoint.rng_state is not None
    assert "scores" in checkpoint.candidates[0]


def test_completed_checkpoint_can_be_resumed(tmp_path):
    checkpoint_path = tmp_path / "darwin-checkpoint.json"
    data = [dspy.Example(question="q", answer="a").with_inputs("question")]
    with dspy.context(lm=DummyLM([{"answer": "a"}])):
        Darwin(
            GEPAStrategy,
            DarwinConfig(max_lm_calls=1, checkpoint_path=str(checkpoint_path)),
        ).compile(dspy.Predict("question -> answer"), trainset=data)

    resumed = Darwin(
        GEPAStrategy,
        DarwinConfig(max_lm_calls=1, resume_from=str(checkpoint_path)),
    )
    compiled = resumed.compile(dspy.Predict("question -> answer"), trainset=data)

    assert compiled._compiled is True
    assert resumed.get_last_result().candidates
    assert type(resumed.strategy.current_newborns).__name__ == "NewBorns"


def test_signal_handler_writes_interrupted_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "signal-checkpoint.json"
    strategy = GEPAStrategy(
        DarwinConfig(max_lm_calls=1, checkpoint_path=str(checkpoint_path))
    )
    strategy._handle_signal(2, None)

    assert strategy._signal_stop_requested is True
    assert strategy._signal_stop_reason == "SIGINT"
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text())
    assert payload["completed"] is False
    assert payload["stop_reason"] == "SIGINT"
