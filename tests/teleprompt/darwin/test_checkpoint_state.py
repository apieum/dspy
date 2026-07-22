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
    assert checkpoint.rng_state is not None
    assert "scores" in checkpoint.candidates[0]
