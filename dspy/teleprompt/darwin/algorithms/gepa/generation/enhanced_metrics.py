"""Enhanced μf assessors for GEPA ReflectivePromptMutation.

These μf (enhanced feedback) assessors demonstrate how to provide rich diagnostic
feedback using the new Metric system that enables intelligent reflection and
prompt mutation. Each assessor IS a μf function that returns enhanced Metric objects.
"""

import ast
import re
import logging
from typing import Optional

import dspy
from ....evaluation.metrics import BaseAssessor, Metric

logger = logging.getLogger(__name__)


class CodeEvaluationAssessor(BaseAssessor):
    """μf assessor that evaluates generated code and provides rich feedback.

    This μf (enhanced feedback function) provides detailed diagnostic information
    using the new Metric system with enhanced feedback capabilities for intelligent
    reflection and prompt mutation.

    Best for:
    - Code generation tasks
    - Programming challenges
    - Algorithm implementation evaluation
    """

    def _evaluate(self, example: dspy.Example, prediction: str, trace=None) -> Metric:
        try:
            # Extract code from prediction (prediction is already normalized to string by base class)
            code = prediction
            if not code.strip():
                return Metric(
                    value=0.0,
                    id=getattr(example, "dspy_uuid", ""),
                    feedback="No code found in prediction output.",
                    suggestions=["Provide a complete code solution", "Include proper Python syntax"]
                )

            # Step 1: Try to parse the code as valid Python
            try:
                ast.parse(code)
            except SyntaxError as e:
                return Metric(
                    value=0.0,
                    id=getattr(example, "dspy_uuid", ""),
                    feedback=f"Syntax error: {e.msg} at line {e.lineno}.",
                    errors={"error_type": "syntax", "line": e.lineno, "message": e.msg},
                    suggestions=[
                        "Check for missing colons after if/for/def statements",
                        "Verify parentheses and brackets are properly matched",
                        "Check indentation consistency"
                    ]
                )
            except Exception as e:
                return Metric(
                    value=0.0,
                    id=getattr(example, "dspy_uuid", ""),
                    feedback=f"Code parsing failed: {str(e)}",
                    errors={"error_type": "parsing", "message": str(e)},
                    suggestions=["Ensure valid Python syntax", "Review basic Python structure"]
                )

            # Step 2: Check for basic function structure if expected
            expected_answer = getattr(example, 'answer', '').lower()
            if 'function' in expected_answer or 'def ' in expected_answer:
                if 'def ' not in code:
                    return Metric(
                        value=0.3,
                        id=getattr(example, "dspy_uuid", ""),
                        feedback="Code compiles but no function definition found. Consider using 'def function_name():' to define a function.",
                        suggestions=["Add function definition using 'def function_name():'"],
                        trace=trace
                    )

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

            # Generate enhanced feedback with suggestions
            suggestions = []
            if quality_score < 1.0:
                suggestions.extend(quality_feedback)
                if 'return' not in code and 'def ' in code:
                    suggestions.append("Add appropriate return statement")
                if quality_score < 0.7:
                    suggestions.append("Review Python best practices")

            # Generate feedback text
            if quality_score >= 0.9:
                feedback = "Code compiles successfully and follows good practices."
            elif quality_score >= 0.7:
                feedback = f"Code compiles but could be improved: {'; '.join(quality_feedback)}."
            else:
                feedback = f"Code has quality issues: {'; '.join(quality_feedback)}."

            return Metric(
                value=quality_score,
                id=getattr(example, "dspy_uuid", ""),
                feedback=feedback,
                errors={"quality_issues": quality_feedback, "score_breakdown": "syntax_ok"},
                suggestions=suggestions,
                trace=trace
            )

        except Exception as e:
            logger.warning(f"Code evaluation assessor failed: {e}")
            return Metric(
                value=0.0,
                id=getattr(example, "dspy_uuid", ""),
                feedback=f"Evaluation failed due to unexpected error: {str(e)}",
                errors={"error_type": "evaluation_failure", "exception": str(e)},
                suggestions=["Check code format and try again"]
            )
