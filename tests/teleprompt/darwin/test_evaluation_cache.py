import dspy

from dspy.teleprompt.darwin import EvaluationCache, Metric


def test_evaluation_cache_reuses_results_for_same_candidate_and_example():
    cache = EvaluationCache()
    candidate = object()
    example = dspy.Example(question="2+2", answer="4").with_inputs("question")
    value = Metric(1.0)

    assert len(cache) == 0
    assert cache.get(candidate, example) is None
    cache.put(candidate, example, value)
    assert cache.get(candidate, example) is value
    assert len(cache) == 1


def test_evaluation_cache_is_scoped_to_candidate_identity():
    cache = EvaluationCache()
    example = dspy.Example(question="2+2", answer="4").with_inputs("question")
    first = object()
    second = object()
    cache.put(first, example, "first")

    assert cache.get(first, example) == "first"
    assert cache.get(second, example) is None
