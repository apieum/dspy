"""Filtering protocol for candidate pool operations."""

from typing import List
from abc import ABC, abstractmethod


class Filtering(ABC):
    """Base protocol for candidate filtering."""
    
    @abstractmethod
    def filter(self, data) -> List[int]:
        """Filter the provided data and return selected candidate IDs."""
        pass