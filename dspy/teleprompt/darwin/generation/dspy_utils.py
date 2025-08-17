"""Common DSPy module operation utilities for GEPA generators.

Consolidates frequently used DSPy module patterns across mutation components.
"""

import logging
from typing import List, Optional, Any, Dict, Tuple
import dspy

logger = logging.getLogger(__name__)


def get_predictors(module: dspy.Module) -> List[Any]:
    """Safely get predictors from a DSPy module.
    
    Args:
        module: DSPy module
        
    Returns:
        List of predictors (empty list if none found)
    """
    try:
        return module.predictors() if module else []
    except Exception as e:
        logger.warning(f"Failed to get predictors from module: {e}")
        return []


def get_predictor_instruction(predictor: Any, default: str = "Answer the question.") -> str:
    """Safely get instruction from a predictor's signature.
    
    Args:
        predictor: DSPy predictor
        default: Default instruction if none found
        
    Returns:
        Instruction text
    """
    try:
        return predictor.signature.instructions or default
    except Exception as e:
        logger.warning(f"Failed to get predictor instruction: {e}")
        return default


def get_valid_predictor(module: dspy.Module, target_idx: int) -> Optional[Any]:
    """Safely get a valid predictor by index from a module.
    
    Args:
        module: DSPy module
        target_idx: Index of target predictor
        
    Returns:
        Predictor if valid, None otherwise
    """
    try:
        predictors = get_predictors(module)
        if not predictors or target_idx >= len(predictors):
            return None
        return predictors[target_idx]
    except (TypeError, AttributeError, IndexError) as e:
        logger.warning(f"Failed to get valid predictor at index {target_idx}: {e}")
        return None


def collect_traces(module: dspy.Module, example: dspy.Example) -> Tuple[Any, List]:
    """Collect execution traces from a DSPy module using native trace collection.
    
    Args:
        module: DSPy module to execute
        example: Example to process
        
    Returns:
        Tuple of (prediction, traces)
    """
    try:
        with dspy.context(trace=[]):
            prediction = module(**example.inputs())
            traces = dspy.settings.trace.copy()
        return prediction, traces
    except Exception as e:
        logger.warning(f"Trace collection failed: {e}")
        return None, []


def validate_module_structure(module: dspy.Module, required_predictors: int = 1) -> bool:
    """Validate that a module has the expected structure.
    
    Args:
        module: DSPy module to validate
        required_predictors: Minimum number of required predictors
        
    Returns:
        True if module structure is valid
    """
    predictors = get_predictors(module)
    return len(predictors) >= required_predictors


class ModuleInspector:
    """Utility class for inspecting DSPy module properties."""
    
    @staticmethod
    def get_all_instructions(module: dspy.Module) -> List[str]:
        """Get all instructions from all predictors in a module.
        
        Args:
            module: DSPy module
            
        Returns:
            List of instruction strings
        """
        instructions = []
        for predictor in get_predictors(module):
            instruction = get_predictor_instruction(predictor)
            instructions.append(instruction)
        return instructions
    
    @staticmethod
    def count_predictors(module: dspy.Module) -> int:
        """Count the number of predictors in a module.
        
        Args:
            module: DSPy module
            
        Returns:
            Number of predictors
        """
        return len(get_predictors(module))
    
    @staticmethod
    def has_predictors(module: dspy.Module) -> bool:
        """Check if module has any predictors.
        
        Args:
            module: DSPy module
            
        Returns:
            True if module has predictors
        """
        return ModuleInspector.count_predictors(module) > 0