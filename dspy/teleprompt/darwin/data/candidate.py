"""Candidate data structure for GEPA optimization."""
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Optional, Union, TYPE_CHECKING
from typing_extensions import Any

from dspy import Module, Example, Prediction

if TYPE_CHECKING:
    from ..evaluation import Assessor, Metric
from ..observers import Channel, NullChannel
from dspy.dsp.utils.settings import settings
#from dspy.primitives.example import Example
from dspy.utils.parallelizer import ParallelExecutor


def example_id(example: Example) -> str:
    """Return a stable task ID for Pareto comparisons.

    Some DSPy versions do not attach ``dspy_uuid`` to ``Example`` instances.
    Falling back to a new UUID at metric construction time would make the same
    task look different on every evaluation and would corrupt GEPA's
    per-example Pareto frontier.  Object identity is stable for the lifetime
    of a compilation and is the correct fallback for those examples.
    """
    return str(
        getattr(example, "dspy_uuid", None)
        or getattr(example, "uuid", None)
        or id(example)
    )


@dataclass
class Candidate:
    """A candidate solution that encapsulates a DSPy module.
    The candidate knows its lineage (parent relationships)
    """
    module: Module  # DSPy module - the actual program
    parents: List['Candidate'] = field(default_factory=list)  # Direct parent references
    generation_number: int = 0  # Which generation this belongs to
    creation_metadata: Dict[str, Any] = field(default_factory=dict)
    proposal_minibatch: Optional[List[Example]] = None
    scores: List["Metric"] = field(default_factory=list)  # List of fitness scores with full context

    def __hash__(self) -> int:
        """Make candidates hashable based on object identity."""
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        """Compare candidates by object identity."""
        if not isinstance(other, Candidate):
            return False
        return self is other

    def find_score_by_uuid(self, example_uuid: str) -> Optional["Metric"]:
        """Find fitness score for a specific example UUID."""
        for score in self.scores:
            if score.id == example_uuid:
                return score
        return None

    def average_score(self) -> float:
        """Calculate average fitness score across all evaluations."""
        if not self.scores:
            return 0.0
        return sum(float(score.value) for score in self.scores) / len(self.scores)

    def evaluate_on_task(self, task: Example, assessor: "Assessor", channel: Optional[Channel] = NullChannel) -> "Metric":
        """Evaluate this candidate on a single example using provided metric."""
        from ..evaluation import Metric

        trace:Dict[str, Any] = {'task': task}
        try:
            prediction = self.module(**task.inputs())
            trace['prediction'] = prediction
            return assessor(task, prediction, trace)
        except Exception as e:
            return Metric(0.0, example_id(task), errors={"evaluate_on_task":e}, trace=trace)  # Failed evaluation

    def evaluate_on_batch(self, examples: List[Example], assessor: "Assessor",
        num_threads=None,
        max_errors=None,
        provide_traceback=False,
        disable_progress_bar=False,
        channel: Optional[Channel] = NullChannel,
        persist_scores: bool = True) -> List["Metric"]:
        """Evaluate this candidate on a batch of examples and return rich Metric objects."""
        from ..evaluation import Metric

        if not examples:
            return []

        executor = ParallelExecutor(
            num_threads=num_threads or settings.num_threads,
            max_errors=max_errors or settings.max_errors,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )

        module = self.module.deepcopy()
        # Get current instruction for trace context
        instruction = "No instruction"
        predictors = module.predictors()
        if predictors:
            from dspy.teleprompt.utils import get_signature
            signature = get_signature(predictors[0])
            instruction = signature.instructions or "No instruction"

        def process_example(example: Example) -> "Metric":
            """Process a single example and return a Metric with rich trace context."""
            try:
                # Make prediction
                prediction = module(**example.inputs())

                # Create rich trace with example UUID and prediction context
                trace = {
                    'example': example,
                    'prediction': prediction,
                    'instruction': instruction,
                    'candidate_id': id(self)
                }

                # Get assessment result
                result = assessor(example, prediction, trace)
                if not isinstance(result, Metric):
                    if isinstance(result, tuple) and len(result) == 2:
                        value, feedback = result
                        result = Metric(value, id=example_id(example), feedback=str(feedback), trace=trace)
                    else:
                        result = Metric(result, id=example_id(example), trace=trace)
                else:
                    # Assessors are allowed to return a Metric, but its
                    # default UUID is not a stable task identifier. Normalize
                    # it before it reaches the Pareto frontier.
                    result = Metric(
                        result.value,
                        id=example_id(example),
                        feedback=result.feedback,
                        errors=result.errors,
                        suggestions=result.suggestions,
                        trace=result.trace if result.trace is not None else trace,
                        objective_scores=result.objective_scores,
                    )

                # Publish evaluation event to observers via channel
                channel.publish('example_evaluated', {
                    'candidate': self,
                    'example': example,
                    'prediction': prediction,
                    'result': result,
                    'instruction': instruction
                })
                return result

            except Exception as e:
                # Create error metric with trace context
                error_trace = {
                    'example': example,
                    'candidate_id': id(self),
                    'error': str(e)
                }

                error_result = Metric(
                    value=0.0,
                    id=example_id(example),
                    feedback=f"Evaluation failed: {str(e)}",
                    errors={'evaluation_error': e},
                    trace=error_trace
                )

                # Publish error event to observers
                channel.publish('evaluation_error', {
                    'candidate': self,
                    'example': example,
                    'error': e
                })

                if settings.max_errors == 0:
                    raise e

                return error_result

        # Execute parallel evaluation and get scores. ParallelExecutor may
        # return ``None`` for an exception it handled internally; normalize
        # that to a failed metric so selection remains well-defined when an LM
        # or budget is exhausted.
        raw_scores = executor.execute(process_example, examples)
        scores = []
        for index, example in enumerate(examples):
            score = raw_scores[index] if raw_scores and index < len(raw_scores) else None
            if isinstance(score, Metric):
                scores.append(score)
            else:
                scores.append(
                    Metric(
                        value=0.0,
                        id=example_id(example),
                        feedback="Evaluation failed before a metric was returned.",
                        trace={"example": example},
                    )
                )

        # Store scores in candidate and publish batch completion
        if scores:
            batch_scores = [s for s in scores if isinstance(s, Metric)]
            if persist_scores:
                self.scores = batch_scores
            if channel:
                channel.publish('evaluate_on_batch', {
                    'candidate': self,
                    'scores': batch_scores,
                    'examples_count': len(examples)
                })

        return batch_scores if scores else []

    def best_overall(self, other: 'Candidate') -> 'Candidate':
        my_avg_score = self.average_score()
        other_avg_score = other.average_score()
        if my_avg_score > other_avg_score:
            return self
        elif my_avg_score < other_avg_score:
            return other
        elif self.generation_number > other.generation_number:
            return self
        else:
            return other

    def dominate(self, other: 'Candidate') -> bool:
        """Check if this candidate Pareto-dominates another candidate.

        Returns True if this candidate performs at least as well on all examples
        and strictly better on at least one example (Pareto dominance).

        Args:
            other: The other candidate to compare against

        Returns:
            True if this candidate dominates the other
        """
        at_least_as_good_on_all = True
        strictly_better_on_one = False

        my_example_uuids = {score.id for score in self.scores}
        other_example_uuids = {score.id for score in other.scores}
        all_example_uuids = my_example_uuids | other_example_uuids

        for example_uuid in all_example_uuids:
            my_score = self.find_score_by_uuid(example_uuid)
            other_score = other.find_score_by_uuid(example_uuid)

            my_value = float(my_score.value) if my_score else 0.0
            other_value = float(other_score.value) if other_score else 0.0

            if my_value < other_value:
                # I'm worse on this example → no domination possible
                at_least_as_good_on_all = False
                break
            elif my_value > other_value:
                # I'm strictly better on this example
                strictly_better_on_one = True
            # else: equal scores → continue checking other examples

        return at_least_as_good_on_all and strictly_better_on_one

    def _get_all_ancestors(self) -> set['Candidate']:
        """
        Private helper to recursively get all unique ancestors.
        This is an internal implementation detail.
        """
        ancestors = set()
        # Use a stack for iterative depth-first traversal to avoid recursion limits
        to_visit = list(self.parents)
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(parent.parents)
        return ancestors

    def is_descendant_of(self, other: 'Candidate'):
        if other in self.parents:
            return True
        for parent in self.parents:
            if parent.is_descendant_of(other):
                return True
        return False

    def is_ancestor_of(self, other: 'Candidate') -> bool:
        """Check if this candidate is an ancestor of another candidate."""
        return other.is_descendant_of(self)

    def filter_ancestors(self, allowed_ancestors: set['Candidate']) -> set['Candidate']:
        """
        Filters this candidate's ancestors, keeping only those present in a given set.
        """
        my_ancestors = self._get_all_ancestors()
        return my_ancestors.intersection(allowed_ancestors)

    def find_common_ancestors(self, other: 'Candidate') -> set['Candidate']:
        """Find all common ancestors shared with another candidate."""
        my_ancestors = self._get_all_ancestors()
        return other.filter_ancestors(my_ancestors)

    def is_ancestor_of_any(self, candidates: List['Candidate']) -> bool:
        """Check if this candidate is an ancestor of any candidate in the given list."""
        return any(candidate.is_descendant_of(self) for candidate in candidates)
