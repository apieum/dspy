from dspy.teleprompt.darwin import CohortModel, DarwinConfig
from dspy.teleprompt.darwin.data.cohort import Cohort


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
