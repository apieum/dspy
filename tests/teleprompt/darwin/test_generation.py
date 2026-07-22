"""Test Darwin generation components (mutation, reflection, merging)."""

import dspy
from dspy.teleprompt.darwin import Bleu, Contains, ExactMatch, F1Score, ReflectiveMutationConfig, RougeL
from dspy.teleprompt.darwin.generation.config import ModuleSelectionStrategy
from dspy.teleprompt.darwin.generation.mutation import ReflectivePromptMutation
from dspy.teleprompt.darwin.generation.feedback import FeedbackProvider
from dspy.teleprompt.darwin.generation.system_aware_merge import SystemAwareMerge
from dspy.teleprompt.darwin.generation.evolvable_module import EvolvableModule
from dspy.teleprompt.darwin.generation.prompt_mutator import ReflectivePromptMutator
from dspy.teleprompt.darwin.evaluation.feedback import FeedbackResult
from dspy.teleprompt.darwin.data.cohort import Parents
from dspy.teleprompt.darwin.data.candidate import Candidate
from dspy.teleprompt.darwin.budget.lm_calls import LMCallsBudget
from dspy.teleprompt.darwin.data.split_strategy import DefaultSplitStrategy
from unittest.mock import Mock, patch


def simple_metric(example, prediction, trace=None):
    return 0.5


class TestGeneration:
    """Test all generation components."""

    def test_mutation_initialization(self):
        """Test mutation generator initialization."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        feedback_data = [dspy.Example(question="q1", answer="a1").with_inputs("question")]
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=feedback_data
        )

        assert generator.feedback_provider == feedback_provider
        assert generator.feedback_data == feedback_data
        assert generator.module_selection == "round_robin"

    def test_evolvable_copy_does_not_read_forward_methods(self, caplog):
        """Wrapping a DSPy predictor must not trigger direct-forward warnings."""
        with caplog.at_level("WARNING", logger="dspy.primitives.module"):
            EvolvableModule(base_module=dspy.Predict("question -> answer"))

        assert "Calling module.forward" not in caplog.text

    def test_reflection_receives_concrete_feedback_by_default(self):
        feedback = FeedbackResult(
            traces=[[(None, {"question": "7 + 5"}, {"answer": "12"})]],
            diagnostics=["Score: 1.00 | Feedback: correct arithmetic"],
            scores=[1.0],
        )
        mutator = ReflectivePromptMutator()

        formatted = mutator._format_feedback_for_reflection(feedback, 0)

        assert "7 + 5" in formatted
        assert "12" in formatted

    def test_mutation_compilation(self):
        """Test mutation generator compilation setup."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        training_data = [
            dspy.Example(question="q1", answer="a1").with_inputs("question"),
            dspy.Example(question="q2", answer="a2").with_inputs("question"),
        ]
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=training_data
        )

        student = dspy.Predict("question -> answer")
        generator.start_compilation(student, verbose=False)

        assert generator.next_module_idx == 0

    def test_mutation_empty_generation(self):
        """Test mutation handles empty cases."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=[]  # Empty feedback data
        )
        budget = LMCallsBudget(100)

        empty_parents = Parents(iteration=0)
        result = generator.generate(empty_parents, budget)
        assert result.is_empty()

    @patch('dspy.teleprompt.darwin.generation.mutation.ReflectivePromptMutator')
    def test_successful_mutation(self, mutator_mock):
        """Test that a successful mutation returns a new candidate."""
        # Arrange
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        feedback_data = [dspy.Example(question="q1", answer="a1").with_inputs("question")]
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=feedback_data
        )
        student_module = dspy.Predict("question -> answer")
        parent_candidate = Candidate(module=student_module)
        parents = Parents(parent_candidate)
        
        mutated_module = dspy.Predict("question -> mutated_answer")
        mutator_instance = mutator_mock.return_value
        mutator_instance.mutate.return_value = mutated_module

        # Act
        new_borns = generator.generate(parents)

        # Assert
        assert not new_borns.is_empty()
        assert len(new_borns) == 1
        child = list(new_borns)[0]
        assert isinstance(child, Candidate)
        assert child.module is mutated_module
        mutator_instance.mutate.assert_called_once()

    def test_round_robin_module_selection(self):
        """Test that the round robin module selection strategy works correctly."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider)
        
        assert generator._select_target_module(3) == 0
        assert generator._select_target_module(3) == 1
        assert generator._select_target_module(3) == 2
        assert generator._select_target_module(3) == 0

    def test_random_module_selection(self):
        """Test that the random module selection strategy works correctly."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider, module_selection="random")
        
        for _ in range(10):
            module_idx = generator._select_target_module(3)
            assert 0 <= module_idx < 3

    def test_reflective_mutation_config_restores_advanced_selection(self):
        """Test restored mutation config is accepted by the new generator path."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        config = ReflectiveMutationConfig(
            module_selection_strategy=ModuleSelectionStrategy.RANDOM,
            max_retries=2,
        )
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider, config=config)

        assert generator.module_selection == "random"
        assert generator.max_retries == 2

    def test_failed_retries_respect_generation_budget(self):
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(
            feedback_provider=feedback_provider,
            feedback_data=[dspy.Example(question="q", answer="a").with_inputs("question")],
            max_retries=5,
        )
        generator._ensure_evolvable = Mock(side_effect=RuntimeError("boom"))
        parent = Candidate(dspy.Predict("question -> answer"))
        budget = LMCallsBudget(4, generation_max_calls=4)

        result = generator.generate(Parents(parent), budget)

        assert result.is_empty()
        assert budget.consumed_calls == 4
        assert budget.consumed_calls <= 4

    def test_richer_metric_exports(self):
        """Test advanced Darwin assessors remain available from the public package."""
        assert ExactMatch is not None
        assert Contains is not None
        assert RougeL is not None
        assert Bleu is not None

    def test_text_metrics_score_exact_and_partial_answers(self):
        example = dspy.Example(answer="The quick brown fox").with_inputs()

        assert ExactMatch()(example, "The quick brown fox").value == 1
        assert Contains()(example, "The quick").value == 1
        assert F1Score()(example, "quick brown fox").value == 0.8571428571428571
        assert RougeL()(example, "The brown fox").value == 0.8571428571428571

    def test_bleu_rewards_overlap_and_penalizes_short_outputs(self):
        example = dspy.Example(answer="the quick brown fox").with_inputs()
        bleu = Bleu()

        assert bleu(example, "the quick brown fox").value == 1.0
        assert bleu(example, "the fox").value == 0.5
        assert bleu(example, "completely unrelated").value == 0.0

    def test_system_aware_merge_initialization(self):
        """Test system aware merge initialization."""
        generator = SystemAwareMerge()
        # Just test that it can be created
        assert generator is not None

    def test_feedback_provider(self):
        """Test feedback provider basic functionality."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        
        example = dspy.Example(question="test", answer="answer")
        prediction = Mock()
        prediction.answer = "answer"
        
        # Test that feedback provider can be created and has the assessor
        assert feedback_provider.assessor == simple_metric
        score = simple_metric(example, prediction)
        assert score == 0.5  # simple_metric always returns 0.5

    def test_feedback_provider_supports_official_gepa_metric_signature(self):
        received = {}

        def metric(gold, pred, trace, pred_name, pred_trace):
            received.update(pred_name=pred_name, pred_trace=pred_trace)
            return 0.25

        provider = FeedbackProvider(assessor=metric)
        trace = [(object(), {"question": "q"}, {"answer": "a"})]
        score, _ = provider.evaluate(
            dspy.Example(question="q", answer="a"),
            Mock(), trace, module_idx=0,
        )

        assert score == 0.25
        assert received["pred_name"] == "0"
        assert received["pred_trace"] == trace[0]

    def test_feedback_side_information_is_preserved(self):
        def metric(example, prediction, trace=None):
            return dspy.teleprompt.darwin.Metric(
                0.5, feedback="diagnostic", side_info={"confidence": 0.8}
            )

        provider = FeedbackProvider(assessor=metric)
        score, diagnostic, side_info = provider.evaluate_rich(
            dspy.Example(question="q", answer="a"), Mock(), []
        )

        assert score == 0.5
        assert "diagnostic" in diagnostic
        assert side_info == {"confidence": 0.8}

    def test_feedback_provider_accepts_short_mu_f_functions(self):
        provider = FeedbackProvider(
            assessor=lambda example, prediction, trace=None: (0.5, "base"),
            feedback_function=lambda example, prediction, trace=None: "additional",
        )
        example = dspy.Example(question="q", answer="a").with_inputs("question")
        score, diagnostic = provider.evaluate(example, dspy.Prediction(answer="a"))
        assert score == 0.5
        assert "additional" in diagnostic


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
