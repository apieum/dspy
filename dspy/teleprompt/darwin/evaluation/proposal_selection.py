"""Selection policies for accepted GEPA proposal batches."""


class AllImprovements:
    """Keep every proposal accepted by the acceptance criterion."""

    def select(self, candidates, improvements):
        return list(candidates)


class BestImprovement:
    """Keep only the proposal with the largest improvement margin."""

    def select(self, candidates, improvements):
        if not candidates:
            return []
        return [candidates[max(range(len(candidates)), key=improvements.__getitem__)]]


class TopKImprovements:
    """Keep at most ``k`` proposals with the largest improvement margins."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def select(self, candidates, improvements):
        order = sorted(range(len(candidates)), key=improvements.__getitem__, reverse=True)
        return [candidates[index] for index in order[: self.k]]
