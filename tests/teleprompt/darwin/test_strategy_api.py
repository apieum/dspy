"""Tests for the refactored Darwin strategy API."""

import pytest

import dspy
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.darwin import (
    Budget,
    Candidate,
    Channel,
    Cohort,
    Darwin,
    GEPAConfig,
    Evaluator,
    Failure,
    FeedbackProvider,
    FullTaskScores,
    FullEvaluationPolicy,
    GEPAStrategy,
    GEPATwoPhasesEval,
    Generator,
    LMCallsBudget,
    Metric,
    ParetoFrontier,
    StrictImprovementAcceptance,
    AllImprovements,
    SingleMutationSampling,
    EpochShuffledBatchSampler,
    ReflectivePromptMutation,
    Selector,
    Success,
)
from dspy.utils.dummies import DummyLM
from dspy.teleprompt.darwin.result import Result
from dspy.teleprompt.darwin.gepa_optimizers import GEPAMute


class SimpleQA(Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


def simple_metric(example, prediction, trace=None):
    expected = example.answer.lower() if hasattr(example, "answer") else ""
    actual = prediction.answer.lower() if hasattr(prediction, "answer") else ""
    return 1.0 if expected == actual else 0.0


@pytest.fixture
def simple_trainset():
    return [
        Example(question="What is 2+2?", answer="4").with_inputs("question"),
        Example(question="What color is the sky?", answer="blue").with_inputs("question"),
        Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        Example(question="How many legs does a cat have?", answer="4").with_inputs("question"),
        Example(question="What planet do we live on?", answer="Earth").with_inputs("question"),
    ]


@pytest.fixture
def dummy_lm():
    return DummyLM([
        {"answer": "4"},
        {"answer": "blue"},
        {"answer": "Paris"},
        {"answer": "4"},
        {"answer": "Earth"},
        {"response": "Improved instruction: Answer questions accurately and concisely."},
    ])


def test_refactored_darwin_returns_compiled_program(simple_trainset, dummy_lm):
    with dspy.context(lm=dummy_lm):
        student = SimpleQA()
        optimizer = Darwin(GEPAStrategy, GEPAConfig(max_lm_calls=2, minibatch_size=2))
        compiled_module = optimizer.compile(student, trainset=simple_trainset[:1], devset=simple_trainset[1:])
        result = optimizer.get_last_result()

    assert isinstance(result, Success)
    assert isinstance(compiled_module, Module)
    assert compiled_module is not student
    assert compiled_module._compiled is True


def test_refactored_darwin_exposes_strategy_components():
    optimizer = Darwin(GEPAStrategy, GEPAConfig(max_lm_calls=2))
    assert isinstance(optimizer, Darwin)
    assert hasattr(optimizer.strategy, "budget")
    assert hasattr(optimizer.strategy, "selector")
    assert hasattr(optimizer.strategy, "generator")
    assert hasattr(optimizer.strategy, "evaluator")


def test_gepa_mute_is_budget_driven_by_default():
    optimizer = GEPAMute(metric=simple_metric, max_calls=8)
    assert optimizer.config.patience is None
    assert optimizer.config.use_merge is True

    mutation_only = Darwin(GEPAStrategy, GEPAConfig(use_merge=False))
    assert mutation_only.config.use_merge is False


def test_darwin_config_is_budget_driven_by_default():
    assert GEPAConfig().patience is None
    assert GEPAConfig().use_merge is True
    with pytest.raises(ValueError):
        GEPAConfig(patience=-1)


def test_gepa_config_materializes_all_strategy_defaults():
    config = GEPAConfig()

    assert isinstance(config.acceptance_criterion, StrictImprovementAcceptance)
    assert isinstance(config.proposal_selection, AllImprovements)
    assert isinstance(config.validation_policy, FullEvaluationPolicy)
    assert isinstance(config.sampling_strategy, SingleMutationSampling)
    assert isinstance(config.batch_sampler, EpochShuffledBatchSampler)
    assert config.batch_evaluator is not None

    with pytest.raises(ValueError, match="validation_policy must be configured"):
        GEPAConfig(validation_policy=None)


def test_strategy_injects_the_same_config_into_components():
    config = GEPAConfig(
        max_lm_calls=7,
        use_merge=False,
        frontier_type="hybrid",
        reflection_lm="configured-reflection-lm",
        reuse_parent_rollouts=False,
        perfect_score=0.9,
        skip_perfect_score=False,
    )
    strategy = GEPAStrategy(config)

    assert strategy.budget.config is config
    assert strategy.selector.frontier_type == "hybrid"
    assert strategy.generator.config is config
    assert strategy.generator.reflection_lm == "configured-reflection-lm"
    assert strategy.generator.reuse_parent_rollouts is False
    assert strategy.generator.perfect_score == 0.9
    assert strategy.generator.skip_perfect_score is False
    assert strategy.crossover_generator.config is config
    assert strategy.crossover_generator.val_overlap_floor == config.merge_val_overlap_floor
    assert strategy.crossover_generator.max_attempts == config.merge_pair_attempts


def test_strategy_minibatch_sampling_uses_seeded_rng():
    first = GEPAStrategy(GEPAConfig(seed=17))
    second = GEPAStrategy(GEPAConfig(seed=17))
    data = list(range(10))

    assert first._create_minibatch(data, 4) == second._create_minibatch(data, 4)


def test_darwin_algorithm_phases(simple_trainset, dummy_lm):
    """The Darwin strategy should execute its optimization phases."""
    with dspy.context(lm=dummy_lm):
        student = SimpleQA()
        optimizer = Darwin(GEPAStrategy, GEPAConfig(max_lm_calls=2, minibatch_size=2))
        compiled_module = optimizer.compile(
            student,
            trainset=simple_trainset[:1],
            devset=simple_trainset[1:],
        )
        result = optimizer.get_last_result()

    assert isinstance(result, Success)
    assert compiled_module is not None
    assert compiled_module._compiled is True
    assert optimizer.strategy.current_generation >= 0


def test_darwin_factory_configuration_creates_optimizer():
    """A Darwin configuration should construct a usable strategy instance."""
    optimizer = Darwin(
        GEPAStrategy,
        GEPAConfig(max_lm_calls=2, patience=3, minibatch_size=2),
    )

    assert isinstance(optimizer, Darwin)
    assert hasattr(optimizer, "strategy")
    assert hasattr(optimizer.strategy, "budget")
    assert hasattr(optimizer.strategy, "selector")
    assert hasattr(optimizer.strategy, "generator")
    assert hasattr(optimizer.strategy, "evaluator")


def test_optimizer_can_be_reused_without_stale_compilation_state(simple_trainset, dummy_lm):
    reusable_lm = DummyLM([{"answer": "4"}] * 100)
    with dspy.context(lm=reusable_lm):
        optimizer = Darwin(GEPAStrategy, GEPAConfig(max_lm_calls=2, max_iterations=1))
        first = optimizer.compile(SimpleQA(), trainset=simple_trainset[:1], devset=simple_trainset[1:])
        second = optimizer.compile(SimpleQA(), trainset=simple_trainset[:1], devset=simple_trainset[1:])

    assert first._compiled is True
    assert second._compiled is True
    assert optimizer.strategy.current_generation <= 1


def test_gepa_result_uses_selector_final_candidate():
    """The final result should use accumulated Pareto state when available."""
    strategy = GEPAStrategy(GEPAConfig(max_lm_calls=1))
    strategy.student = SimpleQA()
    generation_candidate = Candidate(strategy.student.deepcopy())
    selector_candidate = Candidate(strategy.student.deepcopy())
    strategy.best_candidate = generation_candidate

    class SelectorWithFinalCandidate:
        def best_candidate(self):
            return selector_candidate

    strategy._selector = SelectorWithFinalCandidate()
    result = strategy.terminate_compilation()

    assert isinstance(result, Result)
    assert result.get_best_generalist() is selector_candidate


def test_gepa_result_retains_pareto_candidates_and_lineage():
    strategy = GEPAStrategy(GEPAConfig(max_lm_calls=1))
    strategy.student = SimpleQA()
    parent = Candidate(strategy.student.deepcopy())
    child = Candidate(strategy.student.deepcopy(), parents=[parent], generation_number=1)
    parent.scores = [Metric(0.5, id="task")]
    child.scores = [Metric(1.0, id="task")]
    strategy.best_candidate = parent

    class SelectorWithPopulation:
        task_wins = {parent: 0, child: 1}
        example_best_candidates = {"task": [child]}

        def best_candidate(self):
            return child

    strategy._selector = SelectorWithPopulation()
    result = strategy.terminate_compilation()

    assert result.candidates == [child, parent]
    assert result.parents[child] == [parent]
    assert result.val_subscores[child] == {"task": 1.0}
    assert result.per_val_instance_best_candidates["task"] == {child}
