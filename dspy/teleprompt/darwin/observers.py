"""Observer protocols for Darwin components with async event bus."""

from abc import ABC
from typing import Protocol, Dict, List, Any, Callable, TYPE_CHECKING
import asyncio
import copy
import logging
from contextlib import suppress

if TYPE_CHECKING:
    from .data.candidate import Candidate
    from .data.cohort import Cohort, NewBorns, Survivors, Parents
    from .evaluation import Metric
    from .result import Result

logger = logging.getLogger("ChannelContext")

import dspy

class ChannelContext:
    def __init__(self, channel:str):
        self.channel = channel

    def __enter__(self):
        return self.execute

    def __exit__(self, exc_type, exc, tb):
        return False  # don’t suppress exceptions from the block itself

    def execute(self, observer, data, *args, **kwargs):
        method = getattr(observer, self.channel)
        task = asyncio.create_task(method(data, *args, **kwargs))
        task.add_done_callback(self._log_done)

    @staticmethod
    def _log_done(task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info(f"Task: {task} was cancelled")
        except Exception as e:
            logger.exception("Observer task failed", exc_info=e)


class OptimizerObserver(Protocol):
    """Observer for Darwin optimizer lifecycle events."""

    async def start_compilation(self, student: dspy.Module, strategy: 'BaseStrategy') -> None:
        """Called when compilation begins."""
        ...

    async def finish_compilation(self, result: "Result[dspy.Module]") -> None:
        """Called when compilation ends with result."""
        ...

    async def start_iteration(self, iteration: int, cohort: "Cohort") -> None:
        """Called at the start of each optimization iteration."""
        ...

    async def finish_iteration(self, iteration: int, cohort: "Cohort") -> None:
        """Called at the end of each optimization iteration."""
        ...


class SelectorObserver(Protocol):
    """Observer for selector events."""

    async def promote(self, survivors: "Survivors") -> None:
        """Called when candidates are promoted to next generation."""
        ...

    async def update_score(self, candidate: "Candidate", score: "Metric") -> None:
        """Called when a single candidate fitness score is available."""
        ...

    async def update_scores_batch(self, candidates: "Survivors") -> None:
        """Called when multiple candidate fitness scores are available."""
        ...


class GeneratorObserver(Protocol):
    """Observer for generator events."""

    async def generate(self, parents: "Parents", newborns: "NewBorns") -> None:
        """Called when new candidates are generated."""
        ...

    async def filter_candidates(self, candidates: List["Candidate"]) -> None:
        """Called when candidates are filtered."""
        ...

    async def update_instruction(self, old_instruction: str, new_instruction: str, candidate: "Candidate") -> None:
        """Called when a candidate's instruction is updated."""
        ...


class EvaluatorObserver(Protocol):
    """Observer for evaluator events."""

    async def evaluate(self, candidates: "NewBorns", results: "Survivors") -> None:
        """Called when candidates are evaluated."""
        ...


class CandidateObserver(Protocol):
    """Observer for evaluator events."""

    async def evaluate_on_batch(self, candidate: "Candidate", scores: List["Metric"]) -> None:
        """Called when candidate fitness scores are computed."""
        ...

class Channel:
    """Mixin to add observer support to components."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observers: Dict[str, List[Callable]] = {}

    def subscribe(self, observer, event:str) -> None:
        """Add an observer to this component."""
        method = getattr(observer, event, None)
        if method is None:
            raise AttributeError(f"Unable to suscribe: Observer {observer} does not implement the method {event}")
        event_list = self.observers.get(event, [])
        event_list.append(method)
        self.observers[event] = event_list

    def unsubscribe(self, observer, event:str) -> None:
        """Remove an observer from this component."""
        method = getattr(observer, event, None)
        event_list = self.observers.get(event, [])
        if method is None:
            raise AttributeError(f"Unable to unsuscribe: Observer {observer} does not implement the method {event}")
        if method in event_list:
            self.observers[event].remove(method)
        else:
            raise AttributeError(f"Unable to unsuscribe: Observer method {method} not found in the list of observers.")

    def publish(self, method_name: str, data: Any, *args, **kwargs) -> None:
        """Notify observers with async event bus."""
        observers = self.observers.get(method_name, [])
        with ChannelContext(method_name) as notify:
            data_copy = copy.deepcopy(data)
            for observer in observers:
                notify(observer, data_copy, *args, **kwargs)

NullChannel = type("NullChannel", (Channel,), {"__getattr__": lambda self, name: self,
                                        "__call__": lambda self, *a, **kw: self,
                                        "__bool__": lambda self: False,
                                        "__repr__": lambda self: "<NULL_CHANNEL>"})()
