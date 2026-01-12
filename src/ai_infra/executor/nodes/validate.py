"""Pre-write validation for generated code.

Phase 1.1: Validate code BEFORE writing to disk to catch syntax errors instantly.

This module provides:
- ValidationResult: Structured result with error details and repair prompts
- validate_python_code(): AST-based Python syntax validation
- validate_json(): JSON syntax validation
- validate_yaml(): YAML syntax validation
- validate_code_node(): Graph node for pre-write validation

Benefits:
- Catches syntax errors in <1ms (no disk write, no test run)
- Generates targeted repair prompts with line numbers
- Works with any LLM model (validation is deterministic)
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.state import ExecutorGraphState

logger = get_logger("executor.nodes.validate")


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class ValidationResult:
    """Result of code validation.

    Attributes:
        valid: Whether the code passed validation.
        error_type: Type of error if invalid (syntax, indent, encoding, json, yaml).
        error_message: Human-readable error message.
        error_line: Line number where error occurred (1-indexed).
        error_col: Column number where error occurred (1-indexed).
        error_context: Surrounding code context for repair prompts.
    """

    valid: bool
    error_type: Literal["syntax", "indent", "encoding", "json", "yaml", None] = None
    error_message: str | None = None
    error_line: int | None = None
    error_col: int | None = None
    error_context: str | None = None

    @property
    def repair_prompt(self) -> str | None:
        """Generate a targeted repair prompt for the LLM.

        Returns:
            A specific prompt guiding the LLM to fix the error,
            or None if the code is valid.
        """
        if self.valid:
            return None

        parts = [f"Fix the {self.error_type} error"]

        if self.error_line:
            parts.append(f" on line {self.error_line}")
        if self.error_col:
            parts.append(f", column {self.error_col}")

        parts.append(":\n")
        parts.append(f"Error: {self.error_message}\n\n")

        if self.error_context:
            parts.append(f"Context:\n{self.error_context}\n\n")

        parts.append("Return ONLY the corrected code, nothing else.")

        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "valid": self.valid,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_line": self.error_line,
            "error_col": self.error_col,
            "error_context": self.error_context,
            "repair_prompt": self.repair_prompt,
        }


# =============================================================================
# Python Validation
# =============================================================================


def validate_python_code(code: str, filename: str = "<string>") -> ValidationResult:
    """Validate Python code syntax using ast.parse.

    This is instantaneous (<1ms) and catches:
    - Syntax errors (missing colons, brackets, quotes)
    - Indentation errors
    - Invalid escape sequences
    - Encoding issues

    Args:
        code: Python source code to validate.
        filename: Filename for error messages (default: "<string>").

    Returns:
        ValidationResult with error details if invalid.

    Examples:
        >>> result = validate_python_code("def add(a, b):\\n    return a + b")
        >>> result.valid
        True

        >>> result = validate_python_code("def add(a, b)\\n    return a + b")
        >>> result.valid
        False
        >>> result.error_type
        'syntax'
        >>> result.error_line
        1
    """
    if not code or not code.strip():
        return ValidationResult(
            valid=False,
            error_type="syntax",
            error_message="Empty or whitespace-only code",
            error_line=1,
        )

    try:
        ast.parse(code, filename=filename)
        return ValidationResult(valid=True)

    except SyntaxError as e:
        # Extract context around the error
        context = _get_error_context(code, e.lineno) if e.lineno else None

        return ValidationResult(
            valid=False,
            error_type="syntax",
            error_message=e.msg,
            error_line=e.lineno,
            error_col=e.offset,
            error_context=context,
        )

    except IndentationError as e:
        context = _get_error_context(code, e.lineno) if e.lineno else None

        return ValidationResult(
            valid=False,
            error_type="indent",
            error_message=e.msg,
            error_line=e.lineno,
            error_col=e.offset,
            error_context=context,
        )

    except ValueError as e:
        # Encoding errors, null bytes, etc.
        return ValidationResult(
            valid=False,
            error_type="encoding",
            error_message=str(e),
            error_line=1,
        )


def _get_error_context(code: str, error_line: int | None, context_lines: int = 3) -> str:
    """Extract code context around an error line.

    Args:
        code: Full source code.
        error_line: Line number of the error (1-indexed).
        context_lines: Number of lines before/after to include.

    Returns:
        Formatted context string with line numbers and error marker.
    """
    if error_line is None:
        return ""

    lines = code.splitlines()
    start = max(0, error_line - context_lines - 1)
    end = min(len(lines), error_line + context_lines)

    result = []
    for i in range(start, end):
        line_num = i + 1
        marker = ">>> " if line_num == error_line else "    "
        result.append(f"{marker}{line_num:4d} | {lines[i]}")

    return "\n".join(result)


# =============================================================================
# JSON Validation
# =============================================================================


def validate_json(content: str, filename: str = "<string>") -> ValidationResult:
    """Validate JSON syntax.

    Args:
        content: JSON content to validate.
        filename: Filename for error messages.

    Returns:
        ValidationResult with error details if invalid.
    """
    if not content or not content.strip():
        return ValidationResult(
            valid=False,
            error_type="json",
            error_message="Empty or whitespace-only JSON",
            error_line=1,
        )

    try:
        json.loads(content)
        return ValidationResult(valid=True)

    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            error_type="json",
            error_message=e.msg,
            error_line=e.lineno,
            error_col=e.colno,
            error_context=_get_error_context(content, e.lineno),
        )


# =============================================================================
# YAML Validation
# =============================================================================


def validate_yaml(content: str, filename: str = "<string>") -> ValidationResult:
    """Validate YAML syntax.

    Args:
        content: YAML content to validate.
        filename: Filename for error messages.

    Returns:
        ValidationResult with error details if invalid.
    """
    if not content or not content.strip():
        return ValidationResult(
            valid=False,
            error_type="yaml",
            error_message="Empty or whitespace-only YAML",
            error_line=1,
        )

    try:
        # Import here to avoid hard dependency
        import yaml

        yaml.safe_load(content)
        return ValidationResult(valid=True)

    except ImportError:
        # YAML not installed, skip validation
        logger.warning("PyYAML not installed, skipping YAML validation")
        return ValidationResult(valid=True)

    except yaml.YAMLError as e:
        line = None
        col = None
        if hasattr(e, "problem_mark") and e.problem_mark:
            line = e.problem_mark.line + 1
            col = e.problem_mark.column + 1

        return ValidationResult(
            valid=False,
            error_type="yaml",
            error_message=str(e),
            error_line=line,
            error_col=col,
            error_context=_get_error_context(content, line) if line else None,
        )


# =============================================================================
# File Type Detection
# =============================================================================


def _get_validator_for_file(
    file_path: str,
) -> tuple[str, type[ValidationResult] | None]:
    """Get the appropriate validator for a file type.

    Args:
        file_path: Path to the file.

    Returns:
        Tuple of (file_type, validator_function or None).
    """
    path_lower = file_path.lower()

    if path_lower.endswith(".py"):
        return ("python", validate_python_code)
    elif path_lower.endswith(".json"):
        return ("json", validate_json)
    elif path_lower.endswith((".yml", ".yaml")):
        return ("yaml", validate_yaml)
    else:
        # No validation for unknown file types
        return ("unknown", None)


# =============================================================================
# Graph Node
# =============================================================================


async def validate_code_node(state: ExecutorGraphState) -> dict[str, Any]:
    """Validate generated code before writing to disk.

    This node runs AFTER execute_task but BEFORE writing files.
    If validation fails, it sets needs_repair=True and populates
    validation_errors with targeted repair prompts.

    Phase 1.1.2: Pre-write validation node.

    Args:
        state: Current graph state with generated_code dict.

    Returns:
        Updated state with validation results:
        - validated=True if all code passed
        - needs_repair=True if any validation failed
        - validation_errors={file_path: error_dict} for failures

    State Input:
        generated_code: dict[str, str]  # file_path -> code content

    State Output:
        validated: bool
        needs_repair: bool
        validation_errors: dict[str, dict]
        repair_count: int (preserved from input)
    """
    generated_code: dict[str, str] = state.get("generated_code", {})

    if not generated_code:
        logger.debug("No generated code to validate")
        return {
            "validated": True,
            "needs_repair": False,
            "validation_errors": {},
        }

    validation_errors: dict[str, dict[str, Any]] = {}
    all_valid = True

    for file_path, code in generated_code.items():
        file_type, validator = _get_validator_for_file(file_path)

        if validator is None:
            logger.debug(f"Skipping validation for {file_type} file: {file_path}")
            continue

        logger.debug(f"Validating {file_type} file: {file_path}")
        result = validator(code, filename=file_path)

        if not result.valid:
            all_valid = False
            validation_errors[file_path] = result.to_dict()
            logger.warning(
                f"Validation failed for {file_path}: "
                f"{result.error_type} error on line {result.error_line}: "
                f"{result.error_message}"
            )
        else:
            logger.debug(f"Validation passed for {file_path}")

    if all_valid:
        logger.info(f"All {len(generated_code)} file(s) passed validation")
        return {
            "validated": True,
            "needs_repair": False,
            "validation_errors": {},
        }
    else:
        logger.warning(
            f"Validation failed for {len(validation_errors)}/{len(generated_code)} file(s)"
        )
        return {
            "validated": False,
            "needs_repair": True,
            "validation_errors": validation_errors,
            # Preserve repair_count from state (don't increment here)
            "repair_count": state.get("repair_count", 0),
        }
