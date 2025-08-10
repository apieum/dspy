"""Enhanced μf-compliant metrics for GEPA ReflectivePromptMutation.

These metrics demonstrate how to provide rich, diagnostic feedback that enables
much more intelligent reflection and prompt mutation.
"""

import ast
import re
import logging
from typing import Tuple, Optional

import dspy

logger = logging.getLogger(__name__)


def code_evaluation_metric(example: dspy.Example, prediction, trace: Optional[list] = None) -> Tuple[float, str]:
    """
    μf-compliant metric that evaluates generated code and provides rich feedback.
    
    This metric demonstrates how to provide detailed diagnostic information
    that enables the reflection LLM to understand WHY code failed and HOW to fix it.
    
    Returns:
        (score, diagnostic_feedback) tuple for μf compliance
    """
    try:
        # Extract code from prediction
        code = getattr(prediction, 'code', str(prediction))
        if not code or not isinstance(code, str):
            return (0.0, "No code found in prediction output.")
        
        # Step 1: Try to parse the code as valid Python
        try:
            ast.parse(code)
        except SyntaxError as e:
            return (0.0, f"Syntax error: {e.msg} at line {e.lineno}. "
                         f"Check for missing colons, parentheses, or indentation issues.")
        except Exception as e:
            return (0.0, f"Code parsing failed: {str(e)}. Ensure valid Python syntax.")
        
        # Step 2: Check for basic function structure if expected
        expected_answer = getattr(example, 'answer', '').lower()
        if 'function' in expected_answer or 'def ' in expected_answer:
            if 'def ' not in code:
                return (0.3, "Code compiles but no function definition found. "
                           "Consider using 'def function_name():' to define a function.")
        
        # Step 3: Check for common code quality issues
        quality_score = 1.0
        quality_feedback = []
        
        # Check for proper return statements
        if 'def ' in code and 'return ' not in code:
            quality_score -= 0.2
            quality_feedback.append("Function should have a return statement")
            
        # Check for proper parameter usage
        if 'def ' in code and '(' in code and ')' in code:
            # Extract function parameters
            func_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', code)
            if func_match:
                params = func_match.group(1).strip()
                if params and not any(p.strip() in code[func_match.end():] for p in params.split(',')):
                    quality_score -= 0.1
                    quality_feedback.append("Function parameters should be used in the function body")
        
        # Generate feedback
        if quality_score >= 0.9:
            feedback = "Code compiles successfully and follows good practices."
        elif quality_score >= 0.7:
            feedback = f"Code compiles but could be improved: {'; '.join(quality_feedback)}."
        else:
            feedback = f"Code has quality issues: {'; '.join(quality_feedback)}."
        
        return (quality_score, feedback)
        
    except Exception as e:
        logger.warning(f"Code evaluation metric failed: {e}")
        return (0.0, f"Evaluation failed due to unexpected error: {str(e)}")


def math_problem_metric(example: dspy.Example, prediction, trace: Optional[list] = None) -> Tuple[float, str]:
    """
    μf-compliant metric for evaluating mathematical problem solving.
    
    Provides detailed feedback about mathematical reasoning and correctness.
    
    Returns:
        (score, diagnostic_feedback) tuple for μf compliance
    """
    try:
        # Get expected and actual answers
        expected = getattr(example, 'answer', '').strip()
        actual = getattr(prediction, 'answer', str(prediction)).strip()
        
        if not expected or not actual:
            return (0.0, "Missing expected or actual answer for comparison.")
        
        # Try to extract numerical values
        def extract_number(text):
            # Look for numbers in the text
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                try:
                    return float(numbers[-1])  # Take the last number found
                except ValueError:
                    pass
            return None
        
        expected_num = extract_number(expected)
        actual_num = extract_number(actual)
        
        if expected_num is None:
            return (0.0, "Could not extract numerical answer from expected result.")
        
        if actual_num is None:
            return (0.0, "No numerical answer found. Ensure your response includes a clear numerical result.")
        
        # Check for exact match
        if abs(expected_num - actual_num) < 0.001:
            return (1.0, f"Correct answer: {actual_num}")
        
        # Check for close match (within 5%)
        if expected_num != 0 and abs((expected_num - actual_num) / expected_num) < 0.05:
            return (0.8, f"Answer {actual_num} is close to expected {expected_num} but not exact. "
                        "Check your calculations for precision.")
        
        # Check for right order of magnitude
        if expected_num != 0 and abs((expected_num - actual_num) / expected_num) < 0.5:
            return (0.4, f"Answer {actual_num} is in the right ballpark compared to {expected_num}, "
                        "but the calculation appears to have significant errors.")
        
        # Completely wrong
        return (0.0, f"Answer {actual_num} is incorrect (expected {expected_num}). "
                    "Review the problem statement and recalculate step by step.")
        
    except Exception as e:
        logger.warning(f"Math problem metric failed: {e}")
        return (0.0, f"Evaluation failed: {str(e)}")


def text_classification_metric(example: dspy.Example, prediction, trace: Optional[list] = None) -> Tuple[float, str]:
    """
    μf-compliant metric for text classification tasks.
    
    Provides detailed feedback about classification accuracy and reasoning.
    
    Returns:
        (score, diagnostic_feedback) tuple for μf compliance
    """
    try:
        # Extract classifications
        expected = getattr(example, 'label', getattr(example, 'answer', '')).strip().lower()
        actual = getattr(prediction, 'classification', 
                        getattr(prediction, 'label',
                               getattr(prediction, 'answer', str(prediction)))).strip().lower()
        
        if not expected or not actual:
            return (0.0, "Missing expected or actual classification labels.")
        
        # Exact match
        if expected == actual:
            return (1.0, f"Correct classification: '{actual}'")
        
        # Check for partial matches or common variations
        common_mappings = {
            'positive': ['pos', 'good', 'favorable', '+'],
            'negative': ['neg', 'bad', 'unfavorable', '-'],
            'neutral': ['neut', 'neither', 'balanced'],
            'true': ['yes', 't', '1'],
            'false': ['no', 'f', '0']
        }
        
        # Check if actual classification maps to expected
        for canonical, variations in common_mappings.items():
            if expected == canonical and actual in variations:
                return (0.8, f"Classification '{actual}' is correct but use standard format '{expected}'")
            elif expected in variations and actual == canonical:
                return (0.8, f"Classification '{actual}' is correct but expected format was '{expected}'")
        
        # Check for opposite classifications (common error pattern)
        opposites = {
            'positive': 'negative', 'negative': 'positive',
            'true': 'false', 'false': 'true',
            'yes': 'no', 'no': 'yes'
        }
        
        if expected in opposites and actual == opposites[expected]:
            return (0.0, f"Classification '{actual}' is the opposite of expected '{expected}'. "
                        "Review the input text more carefully for sentiment/logic.")
        
        # Completely wrong
        available_classes = set([expected] + [k for k in common_mappings.keys()])
        return (0.0, f"Incorrect classification '{actual}' (expected '{expected}'). "
                    f"Available classes appear to be: {sorted(available_classes)}")
        
    except Exception as e:
        logger.warning(f"Text classification metric failed: {e}")
        return (0.0, f"Evaluation failed: {str(e)}")


def qa_accuracy_metric(example: dspy.Example, prediction, trace: Optional[list] = None) -> Tuple[float, str]:
    """
    μf-compliant metric for question-answering tasks.
    
    Provides detailed feedback about answer quality and completeness.
    
    Returns:
        (score, diagnostic_feedback) tuple for μf compliance  
    """
    try:
        expected = getattr(example, 'answer', '').strip()
        actual = getattr(prediction, 'answer', str(prediction)).strip()
        
        if not expected or not actual:
            return (0.0, "Missing expected or actual answer.")
        
        # Normalize for comparison
        expected_norm = expected.lower()
        actual_norm = actual.lower()
        
        # Exact match
        if expected_norm == actual_norm:
            return (1.0, "Exact answer match")
        
        # Substring match
        if expected_norm in actual_norm:
            return (0.9, f"Answer contains correct information but may be overly verbose. "
                        f"Expected: '{expected}', Got: '{actual}'")
        
        if actual_norm in expected_norm:
            return (0.7, f"Answer is partially correct but incomplete. "
                        f"Expected: '{expected}', Got: '{actual}'")
        
        # Check for key term overlap
        expected_words = set(expected_norm.split())
        actual_words = set(actual_norm.split())
        overlap = expected_words.intersection(actual_words)
        
        if len(overlap) > 0:
            overlap_ratio = len(overlap) / len(expected_words)
            if overlap_ratio > 0.5:
                return (0.6, f"Answer has {len(overlap)}/{len(expected_words)} key terms correct "
                           f"but overall meaning differs. Review the question more carefully.")
            else:
                return (0.3, f"Answer has minimal relevance ({len(overlap)}/{len(expected_words)} key terms). "
                           f"Focus on addressing the specific question asked.")
        
        # No meaningful overlap
        return (0.0, f"Answer appears unrelated to expected response. "
                    f"Expected: '{expected}', Got: '{actual}'. "
                    f"Ensure you understand and directly address the question.")
        
    except Exception as e:
        logger.warning(f"QA accuracy metric failed: {e}")
        return (0.0, f"Evaluation failed: {str(e)}")


