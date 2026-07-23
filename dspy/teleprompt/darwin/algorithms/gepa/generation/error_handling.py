"""Shared error handling utilities for GEPA generators.

Consolidates common exception handling patterns used across mutation components.
"""

import logging
from typing import Any, Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

def with_fallback(fallback_value: T, error_message: str = None) -> Callable[[Callable], Callable]:
    """Decorator that provides fallback value and logging for failed operations.
    
    Args:
        fallback_value: Value to return if the decorated function raises an exception
        error_message: Custom error message prefix (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                prefix = error_message or f"{func_name} failed"
                logger.warning(f"{prefix}: {e}")
                return fallback_value
        return wrapper
    return decorator


def safe_operation(operation: Callable[[], T], 
                  fallback_value: T, 
                  error_message: str = None) -> T:
    """Execute operation safely with fallback value and logging.
    
    Args:
        operation: Function to execute safely
        fallback_value: Value to return if operation fails
        error_message: Custom error message (optional)
        
    Returns:
        Operation result or fallback value
    """
    try:
        return operation()
    except Exception as e:
        message = error_message or "Operation failed"
        logger.warning(f"{message}: {e}")
        return fallback_value


def safe_deepcopy(module: Any, error_message: str = None) -> Any:
    """Safely perform deepcopy operation with fallback.
    
    Args:
        module: Module to deepcopy
        error_message: Custom error message (optional)
        
    Returns:
        Deepcopied module or original module if deepcopy fails
    """
    return safe_operation(
        operation=lambda: module.deepcopy(),
        fallback_value=module,
        error_message=error_message or "Module deepcopy failed"
    )