import dspy

from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.visualization import CandidateTreeVisualizer


def test_candidate_tree_renders_graphviz_dot():
    root = Candidate(dspy.Predict("question -> answer"), generation_number=0)
    child = Candidate(
        dspy.Predict("question -> answer"),
        parents=[root],
        generation_number=1,
    )
    visualizer = CandidateTreeVisualizer()
    visualizer.add_candidate(root, creation_strategy="initial")
    visualizer.add_candidate(child, creation_strategy="mutation")

    dot = visualizer.render_dot()

    assert dot.startswith("digraph DarwinCandidates {")
    assert f"c{id(root)} -> c{id(child)};" in dot
    assert "score=" in dot
