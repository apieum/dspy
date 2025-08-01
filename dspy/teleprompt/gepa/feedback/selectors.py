"""Module selection strategies for targeting specific predictors for mutation."""

from collections import defaultdict

import dspy

from .base import ModuleSelector


class RoundRobinModuleSelector(ModuleSelector):
    """Round-robin module selection strategy."""

    def __init__(self):
        self.module_counts = defaultdict(int)

    def select_module(self, program: dspy.Module) -> int:
        """Select module using round-robin strategy."""
        predictors = program.predictors()
        if not predictors:
            return 0

        # Find the module with the lowest selection count
        program_id = id(program)
        min_count = min(self.module_counts[program_id, i] for i in range(len(predictors)))

        # Among modules with min count, select the first one
        for i in range(len(predictors)):
            if self.module_counts[program_id, i] == min_count:
                self.module_counts[program_id, i] += 1
                return i

        # Fallback (should never reach here)
        return 0