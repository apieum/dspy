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
from dspy.teleprompt.darwin.config import DarwinConfig
from dspy.teleprompt.darwin.strategy.gepa import GEPAStrategy
from unittest.mock import Mock, patch


def simple_metric(example, prediction, trace=None):
    return 0.5


class TraceableTwoPredictorModule(dspy.Module):
    """Small deterministic module exposing two DSPy predictor traces."""

    def __init__(self):
        super().__init__()
        self.first = dspy.Predict("question -> first")
        self.second = dspy.Predict("question -> second")

    def forward(self, question):
        first = dspy.Prediction(first="first output")
        second = dspy.Prediction(second="second output")
        trace = getattr(dspy.settings, "trace", None)
        if trace is not None:
            trace.append((self.first, {"question": question}, first))
            trace.append((self.second, {"question": question}, second))
        return dspy.Prediction(answer=second.second)


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
        example = dspy.Example(question="7 + 5", answer="12").with_inputs("question")
        feedback = FeedbackResult(
            traces=[[(None, {"question": "7 + 5"}, {"answer": "12"})]],
            diagnostics=["Score: 1.00 | Feedback: correct arithmetic"],
            scores=[1.0],
            examples=[example],
        )
        mutator = ReflectivePromptMutator()

        formatted = mutator._format_feedback_for_reflection(feedback, 0)

        assert "7 + 5" in formatted
        assert "12" in formatted
        assert "Inputs:" in formatted
        assert "Expected Outputs:" in formatted
        assert "Generated Outputs:" in formatted

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

    def test_candidate_selection_strategy_controls_parent_sampling(self):
        provider = FeedbackProvider(assessor=simple_metric)
        weaker = Candidate(dspy.Predict("question -> answer"))
        stronger = Candidate(dspy.Predict("question -> answer"))
        from dspy.teleprompt.darwin.evaluation.metrics import Metric

        weaker.scores = [Metric(0.2, id="task")]
        stronger.scores = [Metric(0.9, id="task")]
        parents = Parents(
            weaker,
            stronger,
            task_wins={weaker: 10, stronger: 1},
        )
        generator = ReflectivePromptMutation(
            feedback_provider=provider,
            candidate_selection_strategy="current_best",
        )

        assert generator._select_parent(parents) is stronger

    def test_top_k_candidate_selection_preserves_pareto_weights(self):
        provider = FeedbackProvider(assessor=simple_metric)
        candidates = [Candidate(dspy.Predict("question -> answer")) for _ in range(6)]
        from dspy.teleprompt.darwin.evaluation.metrics import Metric

        for index, candidate in enumerate(candidates):
            candidate.scores = [Metric(0.1 * (index + 1), id="task")]
        parents = Parents(
            *candidates,
            task_wins={candidate: 1 for candidate in candidates},
        )
        generator = ReflectivePromptMutation(
            feedback_provider=provider,
            candidate_selection_strategy="top_k_pareto",
        )

        selected = generator._select_parent(parents)

        assert selected in candidates[-5:]

    def test_random_module_selection(self):
        """Test that the random module selection strategy works correctly."""
        feedback_provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(feedback_provider=feedback_provider, module_selection="random")
        
        for _ in range(10):
            module_idx = generator._select_target_module(3)
            assert 0 <= module_idx < 3

    def test_failed_only_targets_predictor_with_failed_trace(self):
        generator = ReflectivePromptMutation(
            feedback_provider=FeedbackProvider(assessor=simple_metric),
            module_selection=ModuleSelectionStrategy.FAILED_ONLY.value,
        )
        traces = [[
            (object(), {"x": "ok"}, {"y": "fine"}),
            (object(), {"x": "ok"}, {"y": "FailedPrediction: tool error"}),
        ]]

        assert generator._select_failed_module(traces, 2) == 1

    @patch('dspy.teleprompt.darwin.generation.mutation.ReflectivePromptMutator')
    def test_all_selection_mutates_each_predictor_in_one_child(self, mutator_mock):
        provider = FeedbackProvider(assessor=simple_metric)
        generator = ReflectivePromptMutation(
            feedback_provider=provider,
            feedback_data=[dspy.Example(question="q", answer="a").with_inputs("question")],
            module_selection=ModuleSelectionStrategy.ALL.value,
        )
        parent = Candidate(dspy.Module())
        first = dspy.Predict("question -> first")
        second = dspy.Predict("question -> second")
        parent.module = dspy.Module()
        parent.module.first = first
        parent.module.second = second
        mutated = parent.module.deepcopy()
        mutator_mock.return_value.mutate.return_value = mutated

        result = generator.generate(Parents(parent))

        assert not result.is_empty()
        assert mutator_mock.return_value.mutate.call_count == 2

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
        # The failed attempts are bounded by max_retries but are not
        # financially charged when no LM call completed.
        assert budget.consumed_calls == 0
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

    def test_reflection_lm_is_injected_from_darwin_config(self):
        marker = object()
        strategy = GEPAStrategy(DarwinConfig(reflection_lm=marker))

        generator = strategy._instantiate_generator(ReflectivePromptMutation)

        assert generator.reflection_lm is marker

    def test_parent_rollout_reuse_is_configurable(self):
        strategy = GEPAStrategy(DarwinConfig(reuse_parent_rollouts=False))

        generator = strategy._instantiate_generator(ReflectivePromptMutation)

        assert generator.reuse_parent_rollouts is False

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
            pred_name="answer",
            pred_trace=[trace[0]],
        )

        assert score == 0.25
        assert received["pred_name"] == "answer"
        assert received["pred_trace"] == [trace[0]]

    def test_feedback_uses_named_predictor_and_matching_subtrace(self):
        calls = []

        def metric(gold, pred, trace, pred_name, pred_trace):
            calls.append((pred_name, pred_trace))
            return 0.5, f"feedback for {pred_name}"

        provider = FeedbackProvider(assessor=metric)
        example = dspy.Example(question="q", answer="second output").with_inputs("question")
        evolvable = EvolvableModule(base_module=TraceableTwoPredictorModule())

        feedback = evolvable.collect_traces_and_evaluate_many(
            {0: example}, provider, [0, 1]
        )

        assert [name for name, _ in calls] == ["first", "second"]
        assert all(len(pred_trace) == 1 for _, pred_trace in calls)
        assert feedback[0].diagnostics == ["Score: 0.50 (FAILURE) | Feedback: feedback for first"]
        assert feedback[1].diagnostics == ["Score: 0.50 (FAILURE) | Feedback: feedback for second"]

    @patch("dspy.teleprompt.darwin.generation.mutation.ReflectivePromptMutator")
    def test_all_selection_passes_predictor_specific_feedback(self, mutator_mock):
        calls = []

        def metric(gold, pred, trace, pred_name, pred_trace):
            calls.append(pred_name)
            return 0.5, f"feedback for {pred_name}"

        example = dspy.Example(question="q", answer="second output").with_inputs("question")
        generator = ReflectivePromptMutation(
            feedback_provider=FeedbackProvider(assessor=metric),
            feedback_data=[example],
            module_selection=ModuleSelectionStrategy.ALL.value,
        )
        parent = Candidate(TraceableTwoPredictorModule())
        mutator_mock.return_value.mutate.side_effect = lambda module, feedback, *_args, **_kwargs: module

        result = generator.generate(Parents(parent))

        assert not result.is_empty()
        assert calls == ["first", "second"]
        mutation_calls = mutator_mock.return_value.mutate.call_args_list
        assert len(mutation_calls) == 2
        assert "feedback for first" in mutation_calls[0].args[1].diagnostics[0]
        assert "feedback for second" in mutation_calls[1].args[1].diagnostics[0]

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
