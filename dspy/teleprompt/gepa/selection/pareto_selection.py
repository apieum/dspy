"""Pareto-based candidate selection implementing Algorithm 2."""

import logging
import random
from collections import defaultdict
from typing import List

from .selection import Selection
from ..data.candidate import Candidate
from ..data.score_matrix import ScoreMatrix
from ..data.candidate_pool import CandidatePool

logger = logging.getLogger(__name__)


class ParetoSelection(Selection):
    """Pareto-based candidate selection (Algorithm 2 from GEPA paper)."""

    def __init__(self):
        self.selection_counts = defaultdict(int)

    def filter(self, candidate_pool: CandidatePool, score_matrix: ScoreMatrix) -> List[Candidate]:
        """Select candidates using Pareto-based illumination strategy (Algorithm 2).

        Implementation of Algorithm 2 from GEPA paper:
        1. Get candidates that achieve best score on at least one training task (from ScoreMatrix)
        2. Prune strictly dominated candidates  
        3. Return Pareto frontier candidates
        """
        # Use the ParetoFrontier selector which implements Algorithm 2 correctly
        from .pareto_frontier import ParetoFrontier
        
        pareto_selector = ParetoFrontier()
        return candidate_pool.filter(pareto_selector)

