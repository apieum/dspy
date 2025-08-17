"""Candidate data structure for GEPA optimization."""
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Optional, Union
from typing_extensions import Any

from dspy import Module, Example, Prediction
from dspy.dsp.utils.settings import settings
#from dspy.primitives.example import Example
from dspy.utils.parallelizer import ParallelExecutor


@dataclass
class Candidate:
    """A candidate solution that encapsulates a DSPy module.
    The candidate knows its lineage (parent relationships)
    """
    module: Module  # DSPy module - the actual program
    parents: List['Candidate'] = field(default_factory=list)  # Direct parent references
    generation_number: int = 0  # Which generation this belongs to
    creation_metadata: Dict[str, Any] = field(default_factory=dict)
    task_scores: Dict[int, float] = field(default_factory=dict)  # task_scores[task_id] = score

    def __hash__(self) -> int:
        """Make candidates hashable based on object identity."""
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        """Compare candidates by object identity."""
        if not isinstance(other, Candidate):
            return False
        return self is other

    def task_score(self, task_id: int, default: float = 0.0) -> float:
        """Get score for a specific task."""
        return self.task_scores.get(task_id, default)

    def average_task_score(self) -> float:
        """Calculate average score across all tasks."""
        return sum(self.task_scores.values()) / len(self.task_scores) if self.task_scores else 0.0

    def evaluate_on_task(self, task: Example, metric: Callable) -> Tuple[float, Any, Any]:
        """Evaluate this candidate on a single example using provided metric."""
        try:
            prediction = self.module(**task.inputs())
            result = metric(task, prediction)

            # Handle both μf-compliant metrics (tuple) and regular metrics (float)
            if isinstance(result, tuple):
                score, feedback = result
                score = float(score)
            else:
                score = float(result)  # Regular metric returning just score
                feedback = ""

            return (score, feedback, prediction)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Single task evaluation failed: {e}")
            return (0.0, "", prediction)  # Failed evaluation

    def evaluate_on_batch(self, examples: Dict[int, Example], metric: Callable,
        num_threads=None,
        max_errors=None,
        provide_traceback=False,
        disable_progress_bar=False,
        verbose=False) -> Dict[int, float]:
        """Evaluate this candidate on a batch of examples using DSPy's native Evaluate class."""
        if not examples:
            return {}

        executor = ParallelExecutor(
            num_threads=num_threads or settings.num_threads,
            max_errors=max_errors or settings.max_errors,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )

        module = self.module.deepcopy()

        def process_example(example):
            task_id, task = example
            try:
                # Get instruction for verbose logging
                instruction = "No instruction"
                predictors = module.predictors()
                if predictors:
                    from dspy.teleprompt.utils import get_signature
                    signature = get_signature(predictors[0])
                    instruction = signature.instructions or "No instruction"

                prediction = module(**task.inputs())
                result = metric(task, prediction)

                # Verbose logging: show instruction, question, and answer
                if verbose:
                    import logging
                    logger = logging.getLogger(__name__)
                    question = str(task.inputs().get('question', task.inputs()))[:200] + ("..." if len(str(task.inputs())) > 200 else "")

                    # Extract the actual answer field from prediction
                    answer_text = ""
                    if hasattr(prediction, 'answer') and prediction.answer is not None:
                        answer_text = str(prediction.answer)
                    elif hasattr(prediction, 'response') and prediction.response is not None:
                        answer_text = str(prediction.response)
                    else:
                        # Try to extract answer field dynamically
                        prediction_attrs = [attr for attr in dir(prediction) if not attr.startswith('_') and not callable(getattr(prediction, attr))]
                        output_fields = [attr for attr in prediction_attrs if attr not in ['reasoning', 'rationale', 'completions']]

                        if output_fields:
                            if 'answer' in output_fields:
                                answer_text = str(getattr(prediction, 'answer'))
                            else:
                                answer_text = str(getattr(prediction, output_fields[0]))
                        else:
                            answer_text = str(prediction)

                    answer = answer_text[:200] + ("..." if len(answer_text) > 200 else "")
                    logger.info(f"EVALUATION [Task {task_id}]:")
                    logger.info(f"  Instruction: {instruction}")
                    logger.info(f"  Question: {question}")
                    logger.info(f"  Given Answer: {answer}")
                    logger.info(f"  Expected Answer: {task.get('answer')}")

                # Handle both μf-compliant metrics (tuple) and regular metrics (float)
                if isinstance(result, tuple):
                    score, feedback = result
                    score = float(score)
                    if verbose:
                        logger.info(f"  Score: {score:.3f}, Feedback: {feedback}")
                else:
                    score = float(result)  # Regular metric returning just score
                    feedback = ""
                    if verbose:
                        logger.info(f"  Score: {score:.3f}")

                return (task_id, score, feedback)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Candidate evaluation failed for task {task_id}: {e}")
                score = -1.0
                feedback = ""
            finally:
                return (task_id, score, feedback)

        result = executor.execute(process_example, examples.items())

        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Candidate evaluation done for tasks {examples}: {result}")
        if result:
            return {task_id:score for (task_id, score, feedback) in result}
        return {task_id:-1.0 for task_id in examples.keys()}

    def batch_task_scores(self, examples: Dict[int, Example], metric: Callable,
        num_threads=None,
        max_errors=None,
        provide_traceback=False,
        disable_progress_bar=False,
        verbose=False) -> Dict[int, float]:
            self.task_scores = self.evaluate_on_batch(examples, metric, num_threads, max_errors, provide_traceback, disable_progress_bar, verbose)
            return self.task_scores

    def best_overall(self, other: 'Candidate') -> 'Candidate':
        my_avg_score = self.average_task_score()
        other_avg_score = other.average_task_score()
        if my_avg_score > other_avg_score:
            return self
        elif my_avg_score < other_avg_score:
            return other
        elif self.generation_number > other.generation_number:
            return self
        else:
            return other

    def best_for_task(self, task_id: int, other: 'Candidate') -> 'Candidate':
        """Compare two candidates and return the best one for a specific task.

        Considers both score and generation number (newer wins on ties).

        Args:
            task_id: The task to compare performance on
            other: The other candidate to compare against

        Returns:
            The better candidate for the task (self or other)
        """
        my_score = self.task_score(task_id) or 0.0
        other_score = other.task_score(task_id) or 0.0

        # Better score wins
        if my_score > other_score:
            return self

        # Same score - newer generation wins
        if my_score == other_score and self.generation_number > other.generation_number:
            return self

        return other
    def best_on_task(self, task_id, candidates:List['Candidate']) -> 'Candidate':
        """Compare a candidate with a list of candidates and return the best one for a specific task.

        Considers both score and generation number (newer wins on ties).

        Args:
            task_id: The task to compare performance on
            candidates: The list of candidates to compare against

        Returns:
            The better candidate for the task (self or other)
        """
        best_candidate = self
        for candidate in candidates:
            best_candidate = best_candidate.best_for_task(task_id, candidate)
        return best_candidate

    def dominate(self, other: 'Candidate') -> bool:
        """Check if this candidate Pareto-dominates another candidate.

        Returns True if this candidate performs at least as well on all tasks
        and strictly better on at least one task (Pareto dominance).

        Args:
            other: The other candidate to compare against

        Returns:
            True if this candidate dominates the other
        """
        at_least_as_good_on_all = True
        strictly_better_on_one = False

        # Use the task_scores keys from either candidate (they should have the same tasks)
        all_task_ids = set(self.task_scores.keys()) | set(other.task_scores.keys())

        for task_id in all_task_ids:
            my_score = self.task_score(task_id) or 0.0
            other_score = other.task_score(task_id) or 0.0

            if my_score < other_score:
                # I'm worse on this task → no domination possible
                at_least_as_good_on_all = False
                break
            elif my_score > other_score:
                # I'm strictly better on this task
                strictly_better_on_one = True
            # else: equal scores → continue checking other tasks

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
