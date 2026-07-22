"""EvolvableModule using DSPy's native capabilities with PromptMutator separation."""

import logging
from typing import Dict, List, Optional, Any, Iterable

import dspy
from dspy import Module

from .prompt_mutator import PromptMutator, ReflectivePromptMutator
from .reflection_strategy import GEPAReflection
from ..evaluation.feedback import FeedbackResult
from ..evaluation.metrics import Metric
from ..data.candidate import example_id as task_id

logger = logging.getLogger(__name__)


class EvolvableModule(Module):
    """DSPy Module with evolution capabilities using native DSPy systems.
    
    Handles execution and feedback collection using DSPy's built-in systems,
    while delegating mutation logic to configurable PromptMutator strategies.
    """
    
    def __init__(self, 
                 base_module: Optional[Module] = None,
                 prompt_mutator: Optional[PromptMutator] = None,
                 **kwargs):
        """Initialize EvolvableModule.
        
        Args:
            base_module: Existing DSPy module to wrap (copies all attributes)
            prompt_mutator: Strategy for mutating modules based on feedback
            **kwargs: Additional arguments passed to Module.__init__
        """
        super().__init__(**kwargs)
        
        # Copy from base module if provided (using DSPy's native systems)
        if base_module:
            self._copy_from(base_module)
        
        # Configure mutation strategy (default to reflective)
        self.prompt_mutator = prompt_mutator or ReflectivePromptMutator(GEPAReflection())
        
        # Track evolution
        self._generation_number = 0
        self._evolution_history: List[Dict] = []
    
    def _copy_from(self, other_module: Module):
        """Store a deep-copied module and delegate execution to it."""
        wrapped = other_module
        if isinstance(other_module, EvolvableModule):
            wrapped = other_module._base_module
        self._base_module = wrapped.deepcopy()

    def forward(self, *args, **kwargs):
        """Execute the wrapped DSPy module through its public call API."""
        return self._base_module(*args, **kwargs)
    
    def _named_predictors(self) -> List[tuple[str, Any]]:
        """Return the wrapped program's predictor names and objects.

        ``EvolvableModule`` is only an execution wrapper.  Calling
        ``self.named_predictors()`` exposes the wrapper itself as an extra
        module path (for example ``_base_module``), which is not the name a
        GEPA metric sees when it is called on the original DSPy program.
        """
        module = getattr(self, "_base_module", None)
        if module is None:
            module = self
        return list(module.named_predictors())

    @staticmethod
    def _trace_matches_predictor(entry: Any, predictor: Any) -> bool:
        """Check whether a DSPy trace entry belongs to ``predictor``."""
        if not isinstance(entry, (tuple, list)) or len(entry) < 3:
            return False
        traced_predictor = entry[0]
        if traced_predictor is predictor:
            return True

        traced_signature = getattr(traced_predictor, "signature", None)
        target_signature = getattr(predictor, "signature", None)
        if traced_signature is None or target_signature is None:
            return False
        equals = getattr(traced_signature, "equals", None)
        if callable(equals):
            try:
                return bool(equals(target_signature))
            except Exception:
                return False
        return traced_signature == target_signature

    def _predictor_context(
        self,
        trace: List,
        target_module_idx: int,
    ) -> tuple[str, Any, List]:
        """Resolve the named predictor and its invocation sub-trace.

        A predictor can occur more than once in a trace, so the returned
        ``pred_trace`` is a DSPy trace list containing the matching invocation,
        rather than an invocation selected by positional index.
        """
        named_predictors = self._named_predictors()
        if not 0 <= target_module_idx < len(named_predictors):
            return str(target_module_idx), None, []

        pred_name, predictor = named_predictors[target_module_idx]
        matching_entries = [
            entry for entry in trace
            if self._trace_matches_predictor(entry, predictor)
        ]
        # GEPA's metric contract expects DSPyTrace (a list of invocations).
        # Selecting the first invocation is deterministic; the full trace is
        # still retained separately for the proposer and evaluator.
        return pred_name, predictor, matching_entries[:1]

    def collect_traces_and_evaluate_many(
        self,
        examples: Dict[int, dspy.Example],
        feedback_provider,
        target_module_indices: Iterable[int],
    ) -> Dict[int, FeedbackResult]:
        """Run the parent once and create feedback for each target predictor.

        GEPA captures one rollout and then asks the metric for predictor-level
        feedback using the relevant invocation from that rollout.  Keeping the
        rollout shared is important for multi-predictor programs: it avoids
        changing the search budget while ensuring ``all`` and ``failed_only``
        do not reuse predictor-0 diagnostics for every component.
        """
        target_indices = list(dict.fromkeys(target_module_indices))
        if not target_indices:
            return {}

        per_target = {
            index: {
                "scores": [],
                "diagnostics": [],
                "traces": [],
                "metrics": [],
            }
            for index in target_indices
        }
        rollouts = []

        for example in examples.values():
            try:
                with dspy.context(trace=[]):
                    prediction = self(**example.inputs())
                    trace = dspy.settings.trace.copy() if hasattr(dspy.settings, "trace") else []
                rollouts.append((example, prediction, trace, None))
            except Exception as error:
                rollouts.append((example, None, [], error))

        evaluate = getattr(feedback_provider, "evaluate_rich", None)
        evaluator = evaluate or feedback_provider.evaluate
        failure_score = float(getattr(feedback_provider, "failure_score", 0.0))

        for target_module_idx in target_indices:
            values = per_target[target_module_idx]
            for example, prediction, trace, execution_error in rollouts:
                if execution_error is not None:
                    score = failure_score
                    diagnostic = f"ERROR: {execution_error}"
                    side_info = None
                    metric_trace = []
                else:
                    pred_name, _, pred_trace = self._predictor_context(
                        trace, target_module_idx
                    )
                    try:
                        evaluation = evaluator(
                            example,
                            prediction,
                            trace,
                            target_module_idx,
                            pred_name=pred_name,
                            pred_trace=pred_trace,
                        )
                        if len(evaluation) == 3:
                            score, diagnostic, side_info = evaluation
                        else:
                            score, diagnostic = evaluation
                            side_info = None
                        metric_trace = trace
                    except Exception as error:
                        score = failure_score
                        diagnostic = f"ERROR: {error}"
                        side_info = None
                        metric_trace = trace

                values["scores"].append(float(score))
                values["diagnostics"].append(str(diagnostic or ""))
                values["traces"].append(metric_trace)
                values["metrics"].append(
                    Metric(
                        float(score),
                        id=task_id(example),
                        feedback=str(diagnostic or ""),
                        trace=metric_trace,
                        side_info=side_info,
                    )
                )

        return {
            target_module_idx: FeedbackResult(
                scores=values["scores"],
                diagnostics=values["diagnostics"],
                traces=values["traces"],
                examples=list(examples.values()),
                metrics=values["metrics"],
            )
            for target_module_idx, values in per_target.items()
        }

    def collect_traces_and_evaluate(
        self,
        examples: Dict[int, dspy.Example],
        feedback_provider,
        target_module_idx: int = 0,
    ) -> FeedbackResult:
        """Execute on examples and collect target-predictor feedback."""
        return self.collect_traces_and_evaluate_many(
            examples,
            feedback_provider,
            [target_module_idx],
        )[target_module_idx]

    def evolve(self, 
               feedback: FeedbackResult, 
               target_module_idx: int = 0,
               prompt_mutator: Optional[PromptMutator] = None) -> "EvolvableModule":
        """Create evolved copy using DSPy's native systems and PromptMutator.
        
        Args:
            feedback: Feedback from minibatch execution
            target_module_idx: Index of predictor to mutate
            prompt_mutator: Optional custom mutator (uses self.prompt_mutator if None)
            
        Returns:
            New EvolvableModule with mutations applied
        """
        try:
            # Use provided mutator or default to configured one
            mutator = prompt_mutator or self.prompt_mutator
            
            # Step 1: Apply mutation using PromptMutator (handles DSPy deepcopy internally)
            mutated_module = mutator.mutate(self, feedback, target_module_idx)
            
            # Step 2: Wrap mutated module as EvolvableModule
            evolved_copy = EvolvableModule(
                base_module=mutated_module,
                prompt_mutator=self.prompt_mutator
            )
            
            # Step 3: Update evolution metadata
            evolved_copy._generation_number = self._generation_number + 1
            evolved_copy._evolution_history = self._evolution_history + [{
                'generation': evolved_copy._generation_number,
                'target_module': target_module_idx,
                'avg_score': sum(feedback.scores) / len(feedback.scores) if feedback.scores else 0.0,
                'mutator_type': type(mutator).__name__,
                'mutation_applied': True
            }]
            
            logger.debug(f"Evolved to generation {evolved_copy._generation_number} using {type(mutator).__name__}")
            return evolved_copy
            
        except Exception as e:
            logger.warning(f"Evolution failed: {e}")
            # Return unchanged copy on failure using DSPy's deepcopy
            return EvolvableModule(
                base_module=self.deepcopy(),
                prompt_mutator=self.prompt_mutator
            )
    
    def with_mutator(self, mutator: PromptMutator) -> "EvolvableModule":
        """Create new EvolvableModule with different PromptMutator.
        
        Args:
            mutator: New prompt mutation strategy
            
        Returns:
            New EvolvableModule with updated mutator
        """
        return EvolvableModule(
            base_module=self.deepcopy(),  # DSPy's native deepcopy
            prompt_mutator=mutator
        )
    
    @property
    def generation_number(self) -> int:
        """Get generation number."""
        return self._generation_number
    
    @property
    def evolution_history(self) -> List[Dict]:
        """Get evolution history."""
        return self._evolution_history.copy()
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution summary."""
        return {
            "generation_number": self._generation_number,
            "mutations_applied": len(self._evolution_history),
            "reflection_strategy": type(self.reflection_strategy).__name__,
            "avg_performance": (
                sum(entry['avg_score'] for entry in self._evolution_history) / 
                len(self._evolution_history)
            ) if self._evolution_history else 0.0
        }
