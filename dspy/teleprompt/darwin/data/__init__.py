"""Generic Darwin data structures."""

from .candidate import Candidate, CandidateOperations
from .cohort import Cohort, Survivors, Parents, NewBorns

__all__ = [
    'Candidate',
    'CandidateOperations',
    'Cohort',
    'Survivors',
    'Parents', 
    'NewBorns',
]
