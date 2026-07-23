"""GEPA candidate data and delegated GEPA candidate operations."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import dspy
from dspy.dsp.utils.settings import settings
from dspy.utils.parallelizer import ParallelExecutor

from ...data.candidate import Candidate, CandidateOperations
from ...evaluation import Metric
from ...observers import Channel, NullChannel


def example_id(example: dspy.Example) -> str:
    """Return a stable task identifier for GEPA's per-example frontier."""
    return str(
        getattr(example, "dspy_uuid", None)
        or getattr(example, "uuid", None)
        or id(example)
    )


class GEPACandidateOperations(CandidateOperations[dspy.Module]):
    """GEPA's evaluation and comparison policy.

    This object is deliberately separate from both Darwin's generic
    candidate model and the candidate's lineage data.  Other algorithms can
    inject a different operations object without inheriting GEPA semantics.
    """

    def evaluate(self, candidate: "GEPACandidate", *args: Any, **kwargs: Any) -> Any:
        """Evaluate a candidate on a task or batch.

        The first positional argument selects the existing GEPA operation:
        ``evaluate(candidate, task, assessor, ...)`` or
        ``evaluate(candidate, examples, assessor, ...)``.
        """
        if len(args) < 2:
            raise TypeError("GEPA evaluation requires examples and an assessor")
        examples, assessor = args[:2]
        options = dict(kwargs)
        if isinstance(examples, dspy.Example):
            return self.evaluate_on_task(candidate, examples, assessor, **options)
        return self.evaluate_on_batch(candidate, examples, assessor, **options)

    def compare(
        self,
        left: "GEPACandidate",
        right: "GEPACandidate",
        *args: Any,
        **kwargs: Any,
    ) -> int:
        criterion = kwargs.pop("criterion", "overall")
        if criterion == "pareto":
            left_dominates = self.dominates(left, right)
            right_dominates = self.dominates(right, left)
            return int(left_dominates) - int(right_dominates)
        if criterion == "task":
            if not args:
                raise TypeError("task comparison requires a task identifier")
            left_score = self.score_for_task(left, args[0])
            right_score = self.score_for_task(right, args[0])
            return (left_score > right_score) - (left_score < right_score)

        left_score = self.total_score(left)
        right_score = self.total_score(right)
        if left_score == right_score:
            return (left.generation_number > right.generation_number) - (
                left.generation_number < right.generation_number
            )
        return (left_score > right_score) - (left_score < right_score)

    def find_score_by_uuid(self, candidate: "GEPACandidate", example_uuid: str) -> Optional[Metric]:
        return next((score for score in candidate.scores if score.id == example_uuid), None)

    def score_for_task(self, candidate: "GEPACandidate", example_uuid: str) -> float:
        score = self.find_score_by_uuid(candidate, str(example_uuid))
        return float(score.value) if score is not None else 0.0

    def average_score(self, candidate: "GEPACandidate") -> float:
        return (
            sum(float(score.value) for score in candidate.scores) / len(candidate.scores)
            if candidate.scores
            else 0.0
        )

    def total_score(self, candidate: "GEPACandidate") -> float:
        return sum(float(score.value) for score in candidate.scores)

    def best_overall(self, left: "GEPACandidate", right: "GEPACandidate") -> "GEPACandidate":
        return left if self.compare(left, right) > 0 else right

    def best_for_task(
        self, left: "GEPACandidate", right: "GEPACandidate", task_id: str
    ) -> "GEPACandidate":
        comparison = self.compare(left, right, task_id, criterion="task")
        return left if comparison > 0 or (
            comparison == 0 and left.generation_number > right.generation_number
        ) else right

    def best_on_task(
        self,
        candidate: "GEPACandidate",
        candidates: List["GEPACandidate"],
        task_id: str,
    ) -> "GEPACandidate":
        best = candidate
        for other in candidates:
            best = self.best_for_task(best, other, task_id)
        return best

    def dominates(self, left: "GEPACandidate", right: "GEPACandidate") -> bool:
        task_ids = {score.id for score in left.scores} | {score.id for score in right.scores}
        strictly_better = False
        for task_id in task_ids:
            left_score = self.score_for_task(left, task_id)
            right_score = self.score_for_task(right, task_id)
            if left_score < right_score:
                return False
            strictly_better = strictly_better or left_score > right_score
        return strictly_better

    def evaluate_on_task(
        self,
        candidate: "GEPACandidate",
        task: dspy.Example,
        assessor,
        channel: Optional[Channel] = NullChannel,
    ) -> Metric:
        trace = {"task": task}
        try:
            prediction = candidate.module(**task.inputs())
            trace["prediction"] = prediction
            return assessor(task, prediction, trace)
        except Exception as error:
            return Metric(0.0, example_id(task), errors={"evaluate_on_task": error}, trace=trace)

    def evaluate_on_batch(
        self,
        candidate: "GEPACandidate",
        examples: List[dspy.Example],
        assessor,
        num_threads=None,
        max_errors=None,
        provide_traceback=False,
        disable_progress_bar=False,
        channel: Optional[Channel] = NullChannel,
        persist_scores: bool = True,
    ) -> List[Metric]:
        if not examples:
            return []

        executor = ParallelExecutor(
            num_threads=num_threads or settings.num_threads,
            max_errors=max_errors or settings.max_errors,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )
        module = candidate.module.deepcopy()
        instruction = "No instruction"
        predictors = module.predictors()
        if predictors:
            from dspy.teleprompt.utils import get_signature

            instruction = get_signature(predictors[0]).instructions or "No instruction"

        def process_example(example: dspy.Example) -> Metric:
            try:
                prediction = module(**example.inputs())
                trace = {
                    "example": example,
                    "prediction": prediction,
                    "instruction": instruction,
                    "candidate_id": id(candidate),
                }
                result = assessor(example, prediction, trace)
                if not isinstance(result, Metric):
                    if isinstance(result, tuple) and len(result) == 2:
                        value, feedback = result
                        result = Metric(value, id=example_id(example), feedback=str(feedback), trace=trace)
                    else:
                        result = Metric(result, id=example_id(example), trace=trace)
                else:
                    result = Metric(
                        result.value,
                        id=example_id(example),
                        feedback=result.feedback,
                        errors=result.errors,
                        suggestions=result.suggestions,
                        trace=result.trace if result.trace is not None else trace,
                        objective_scores=result.objective_scores,
                        side_info=result.side_info,
                    )
                channel.publish(
                    "example_evaluated",
                    {"candidate": candidate, "example": example, "prediction": prediction, "result": result, "instruction": instruction},
                )
                return result
            except Exception as error:
                error_result = Metric(
                    value=0.0,
                    id=example_id(example),
                    feedback=f"Evaluation failed: {error}",
                    errors={"evaluation_error": error},
                    trace={"example": example, "candidate_id": id(candidate), "error": str(error)},
                )
                channel.publish("evaluation_error", {"candidate": candidate, "example": example, "error": error})
                if settings.max_errors == 0:
                    raise
                return error_result

        raw_scores = executor.execute(process_example, examples)
        scores = [
            score if isinstance(score, Metric) else Metric(
                value=0.0,
                id=example_id(example),
                feedback="Evaluation failed before a metric was returned.",
                trace={"example": example},
            )
            for example, score in zip(examples, raw_scores or [], strict=False)
        ]
        scores.extend(
            Metric(value=0.0, id=example_id(example), feedback="Evaluation failed before a metric was returned.", trace={"example": example})
            for example in examples[len(scores):]
        )
        if persist_scores:
            candidate.scores = scores
        if channel:
            channel.publish("evaluate_on_batch", {"candidate": candidate, "scores": scores, "examples_count": len(examples)})
        return scores


@dataclass(init=False, eq=False)
class GEPACandidate(Candidate[dspy.Module]):
    """GEPA-specific candidate payload and evaluation evidence."""

    proposal_minibatch: Optional[List[dspy.Example]] = None
    scores: List[Metric] = field(default_factory=list)

    def __init__(
        self,
        module: dspy.Module,
        parents: List["GEPACandidate"] | None = None,
        generation_number: int = 0,
        creation_metadata: dict[str, Any] | None = None,
        proposal_minibatch: Optional[List[dspy.Example]] = None,
        scores: List[Metric] | None = None,
        operations: GEPACandidateOperations | None = None,
    ) -> None:
        super().__init__(
            value=module,
            parents=parents,
            generation_number=generation_number,
            creation_metadata=creation_metadata,
            operations=operations or GEPACandidateOperations(),
        )
        self.proposal_minibatch = proposal_minibatch
        self.scores = list(scores or [])

    @property
    def module(self) -> dspy.Module:
        return self.value

    @module.setter
    def module(self, value: dspy.Module) -> None:
        self.value = value

    @property
    def gepa_operations(self) -> GEPACandidateOperations:
        return self.operations  # type: ignore[return-value]

    def find_score_by_uuid(self, example_uuid: str) -> Optional[Metric]:
        return self.gepa_operations.find_score_by_uuid(self, example_uuid)

    def average_score(self) -> float:
        return self.gepa_operations.average_score(self)

    def total_score(self) -> float:
        return self.gepa_operations.total_score(self)

    def evaluate_on_task(self, task: dspy.Example, assessor, channel: Optional[Channel] = NullChannel) -> Metric:
        return self.gepa_operations.evaluate_on_task(self, task, assessor, channel)

    def evaluate_on_batch(self, examples: List[dspy.Example], assessor, **kwargs: Any) -> List[Metric]:
        return self.gepa_operations.evaluate_on_batch(self, examples, assessor, **kwargs)

    def best_overall(self, other: "GEPACandidate") -> "GEPACandidate":
        return self.gepa_operations.best_overall(self, other)

    def best_for_task(self, task_id: str, other: "GEPACandidate") -> "GEPACandidate":
        return self.gepa_operations.best_for_task(self, other, task_id)

    def best_on_task(self, task_id: str, candidates: List["GEPACandidate"]) -> "GEPACandidate":
        return self.gepa_operations.best_on_task(self, candidates, task_id)

    def dominate(self, other: "GEPACandidate") -> bool:
        return self.gepa_operations.dominates(self, other)
