"""Generic result contract for Darwin strategies."""


class Result:
    """Marker base class for strategy-specific compilation results."""


class OptimizationFailureError(Exception):
    """Raised when a strategy cannot produce a compiled module."""
