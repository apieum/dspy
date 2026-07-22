from dspy.teleprompt.darwin import CohortModel, DarwinConfig
from dspy.teleprompt.darwin.data.cohort import Cohort
from dspy.teleprompt.darwin import ExecutionGraph, GEPAStrategy


class TaggedCohort(Cohort):
    def __init__(self, *candidates, tag=None, **kwargs):
        self.tag = tag
        super().__init__(*candidates, **kwargs)


def test_cohort_model_materializes_configured_roles():
    model = CohortModel(
        cohort_type=TaggedCohort,
        parents_type=TaggedCohort,
        newborns_type=TaggedCohort,
        survivors_type=TaggedCohort,
    )

    newborns = model.newborns([], iteration=3, tag="proposal")
    parents = model.parents([], iteration=4, tag="lineage")

    assert isinstance(newborns, TaggedCohort)
    assert isinstance(parents, TaggedCohort)
    assert newborns.iteration == 3
    assert parents.tag == "lineage"


def test_darwin_config_provides_a_default_cohort_model():
    config = DarwinConfig()

    assert isinstance(config.cohort_model, CohortModel)
    assert config.cohort_model.newborns([]).is_empty()


def test_execution_graph_dispatches_nodes_without_knowing_gepa_phases():
    class Context:
        algorithm_state = "first"
        calls = []

    context = Context()
    graph = ExecutionGraph({
        "first": lambda value: (value.calls.append("first"), setattr(value, "algorithm_state", "done")),
    }, terminal_states=frozenset({"done"}))

    assert graph.step(context)
    assert context.calls == ["first"]
    assert not graph.step(context)


def test_gepa_strategy_accepts_an_execution_graph_factory():
    def graph_factory(strategy):
        return ExecutionGraph.gepa(strategy)

    strategy = GEPAStrategy(DarwinConfig(execution_graph=graph_factory))

    assert isinstance(strategy.execution_graph, ExecutionGraph)
