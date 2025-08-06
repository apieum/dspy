"""Utility functions for GEPA System-Aware Merge algorithm."""

from .ancestry_traversal import (
    find_common_ancestors,
    is_ancestor_of
)

from .signature_complexity import (
    signature_field_counts,
    is_signature_more_complex,
    compare_signature_complexity
)


from .merge_history import (
    MergeAttempt,
    MergeHistoryTracker
)

__all__ = [
    # Ancestry utilities
    'find_common_ancestors', 
    'is_ancestor_of',
    
    # Signature complexity
    'signature_field_counts',
    'is_signature_more_complex',
    'compare_signature_complexity',
    
    # Merge history
    'MergeAttempt',
    'MergeHistoryTracker'
]