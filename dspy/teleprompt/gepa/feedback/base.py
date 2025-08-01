"""Base interfaces for feedback collection and prompt mutation."""

from abc import ABC, abstractmethod
from typing import Callable, List

import dspy
from dspy.signatures.signature import Signature

from ..data.structures import FeedbackResult


class PromptMutator(ABC):
    """Interface for reflective prompt mutation (core GEPA innovation)."""

    @abstractmethod
    def mutate_signature(self, current_signature: Signature, feedback: FeedbackResult) -> Signature:
        """Mutate signature based on reflective feedback."""
        raise NotImplementedError


class FeedbackCollector(ABC):
    """Interface for enhanced feedback collection (μf function)."""

    @abstractmethod
    def collect_feedback(self, program: dspy.Module, examples: List[dspy.Example], metric: Callable) -> FeedbackResult:
        """Collect enhanced feedback with traces and diagnostics."""
        raise NotImplementedError


class ModuleSelector(ABC):
    """Interface for module selection within programs."""

    @abstractmethod
    def select_module(self, program: dspy.Module) -> int:
        """Select module index to mutate."""
        raise NotImplementedError