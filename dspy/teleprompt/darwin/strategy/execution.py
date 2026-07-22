"""Small, composable execution graphs for Darwin strategies."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping


Step = Callable[[Any], None]


@dataclass
class ExecutionGraph:
    """Dispatch strategy phases by state.

    Nodes receive the strategy context and may set its next state.  The graph
    does not know about cohorts, budgets, or GEPA-specific roles; those remain
    owned by the strategy and its components.
    """

    nodes: Mapping[str, Step]
    terminal_states: frozenset[str] = frozenset({"terminate", "completed"})

    def step(self, context: Any) -> bool:
        state = getattr(context, "algorithm_state", None)
        if state in self.terminal_states:
            return False
        try:
            action = self.nodes[state]
        except KeyError as exc:
            raise RuntimeError(f"Execution graph has no node for state {state!r}") from exc
        action(context)
        return True

    @classmethod
    def gepa(cls, strategy: Any) -> "ExecutionGraph":
        """Build the default GEPA phase graph for a strategy instance."""
        return cls({
            "evaluate": lambda _context: strategy._evaluate_step(),
            "select": lambda _context: strategy._select_step(),
            "generate": lambda _context: strategy._generate_step(),
        })
