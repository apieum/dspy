"""Signature complexity measurement for System-Aware Merge algorithm."""

from typing import Union
from dspy.signatures.signature import Signature


def signature_field_counts(signature: Signature) -> tuple[int, int]:
    """Get input and output field counts from a DSPy signature.
    
    Args:
        signature: DSPy signature object
        
    Returns:
        Tuple of (input_count, output_count)
    """
    if signature is None:
        return (0, 0)
    
    try:
        input_count = len(signature.input_fields) if hasattr(signature, 'input_fields') else 0
        output_count = len(signature.output_fields) if hasattr(signature, 'output_fields') else 0
        return (input_count, output_count)
    except Exception:
        return (0, 0)


def is_signature_more_complex(ancestor_sig: Signature, parent1_sig: Signature, parent2_sig: Signature) -> bool:
    """Check if ancestor signature is more complex than both parents.
    
    Implements the paper's condition: S[a] > min(S[i], S[j])
    Where complexity is measured separately for inputs and outputs.
    
    Args:
        ancestor_sig: Ancestor signature to check
        parent1_sig: First parent signature  
        parent2_sig: Second parent signature
        
    Returns:
        True if ancestor is more complex (should be skipped in merge)
    """
    # Get field counts for each signature
    a_inputs, a_outputs = signature_field_counts(ancestor_sig)
    p1_inputs, p1_outputs = signature_field_counts(parent1_sig)
    p2_inputs, p2_outputs = signature_field_counts(parent2_sig)
    
    # Compare inputs: ancestor_inputs > min(parent1_inputs, parent2_inputs)
    min_parent_inputs = min(p1_inputs, p2_inputs)
    inputs_more_complex = a_inputs > min_parent_inputs
    
    # Compare outputs: ancestor_outputs > min(parent1_outputs, parent2_outputs)  
    min_parent_outputs = min(p1_outputs, p2_outputs)
    outputs_more_complex = a_outputs > min_parent_outputs
    
    # Ancestor is "more complex" if BOTH input AND output counts are higher
    return inputs_more_complex and outputs_more_complex


def compare_signature_complexity(sig1: Signature, sig2: Signature) -> int:
    """Compare complexity of two signatures based on total field count.
    
    Args:
        sig1: First signature
        sig2: Second signature
        
    Returns:
        -1 if sig1 < sig2, 0 if equal, 1 if sig1 > sig2
    """
    inputs1, outputs1 = signature_field_counts(sig1)
    inputs2, outputs2 = signature_field_counts(sig2)
    
    total1 = inputs1 + outputs1
    total2 = inputs2 + outputs2
    
    if total1 < total2:
        return -1
    elif total1 > total2:
        return 1
    else:
        return 0