"""Shared instruction updating utility for GEPA generators.

Uses DSPy's native signature system for reliable instruction updates.
"""

import logging
from typing import Any

from dspy.teleprompt.utils import get_signature, set_signature

logger = logging.getLogger(__name__)


class InstructionUpdater:
    """Handles instruction updates using DSPy's native signature system.
    
    This replaces the duplicated _update_predictor_instruction methods
    across different mutators with a single, clean implementation.
    """
    
    @staticmethod
    def update_instruction(predictor: Any, new_instruction: str) -> None:
        """Update predictor's instruction using DSPy's native utilities.
        
        Args:
            predictor: DSPy predictor to update
            new_instruction: New instruction text
        """
        try:
            # Use DSPy's native signature utilities (same as MIPROv2)
            current_signature = get_signature(predictor)
            updated_signature = current_signature.with_instructions(new_instruction)
            set_signature(predictor, updated_signature)
            
        except Exception as e:
            logger.warning(f"Failed to update predictor instruction using DSPy utilities: {e}")