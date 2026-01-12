"""Repair code node for executor graph.

Phase 1.2: Surgically fix validation errors with targeted prompts.

This module provides:
- MAX_REPAIRS: Maximum repair attempts before escalating
- REPAIR_PROMPT_TEMPLATE: Template for surgical repair prompts
- repair_code_node(): Graph node for repairing validation errors
- should_repair(): Helper to check if repair should be attempted

Key Design Decisions:
- Repair is different from Retry: targeted fixes vs full regeneration
- Uses line-specific prompts for surgical precision
- Escalates to retry after MAX_REPAIRS attempts
- Maintains original code structure when possible
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent


logger = get_logger("executor.nodes.repair")


# =============================================================================
# Constants
# =============================================================================

MAX_REPAIRS = 2  # Maximum repair attempts per validation cycle


# =============================================================================
# Repair Prompt Templates
# =============================================================================

REPAIR_PROMPT_TEMPLATE = """Fix the following {error_type} error in this Python code.

**Error Location**: Line {line}
**Error Message**: {message}

**Original Code**:
```python
{code}
```

**Instructions**:
1. Fix ONLY the error mentioned above
2. Do NOT change any other code
3. Return the COMPLETE fixed file
4. Do NOT include markdown code fences in your response
"""

JSON_REPAIR_PROMPT_TEMPLATE = """Fix the following JSON error.

**Error Message**: {message}

**Original JSON**:
```json
{code}
```

**Instructions**:
1. Fix ONLY the JSON syntax error
2. Preserve the data structure
3. Return the COMPLETE fixed JSON
4. Do NOT include markdown code fences in your response
"""

YAML_REPAIR_PROMPT_TEMPLATE = """Fix the following YAML error.

**Error Message**: {message}

**Original YAML**:
```yaml
{code}
```

**Instructions**:
1. Fix ONLY the YAML syntax error
2. Preserve the data structure
3. Return the COMPLETE fixed YAML
4. Do NOT include markdown code fences in your response
"""


def _get_repair_prompt_template(error_type: str) -> str:
    """Get the appropriate prompt template for an error type.

    Args:
        error_type: Type of validation error (syntax, indent, json, yaml, etc.)

    Returns:
        Prompt template string.
    """
    if error_type == "json":
        return JSON_REPAIR_PROMPT_TEMPLATE
    if error_type == "yaml":
        return YAML_REPAIR_PROMPT_TEMPLATE
    # Default to Python repair for syntax, indent, encoding errors
    return REPAIR_PROMPT_TEMPLATE


def _clean_repaired_code(code: str) -> str:
    """Clean up LLM response to extract raw code.

    Removes markdown code fences and extra whitespace.

    Args:
        code: Raw LLM response.

    Returns:
        Cleaned code string.
    """
    cleaned = code.strip()

    # Remove markdown code fences
    # Handle ```python, ```json, ```yaml, etc.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```python or similar)
        lines = lines[1:]
        # Remove last line if it's closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    return cleaned


# =============================================================================
# Repair Node
# =============================================================================


async def repair_code_node(
    state: ExecutorGraphState,
    *,
    agent: Agent | None = None,
) -> ExecutorGraphState:
    """Surgically repair validation errors.

    Phase 1.2: This node uses targeted prompts to fix specific errors,
    rather than regenerating the entire file.

    The repair strategy is surgical:
    - Uses line-specific error information
    - Generates focused prompts that ask for minimal changes
    - Tracks repair attempts to prevent infinite loops
    - Escalates to full retry after MAX_REPAIRS attempts

    Args:
        state: Current state with validation_errors.
        agent: Agent for LLM calls (required for actual repairs).

    Returns:
        Updated state with repaired code, or error if max repairs exceeded.

    Examples:
        >>> state = {"validation_errors": {...}, "generated_code": {...}}
        >>> updated = await repair_code_node(state, agent=my_agent)
        >>> updated["repair_count"]
        1
    """
    validation_errors: dict = state.get("validation_errors", {})
    generated_code: dict = state.get("generated_code", {})
    repair_count: int = state.get("repair_count", 0)

    # Get task ID for error tracking
    current_task = state.get("current_task")
    task_id = str(current_task.id) if current_task else None

    # Check if max repairs exceeded
    if repair_count >= MAX_REPAIRS:
        logger.error(f"Max repairs ({MAX_REPAIRS}) exceeded, escalating to retry")
        return {
            **state,
            "needs_repair": False,
            "validated": False,
            "error": {
                "error_type": "validation",
                "message": f"Failed to repair after {MAX_REPAIRS} attempts",
                "node": "repair_code",
                "task_id": task_id,
                "recoverable": True,
                "stack_trace": None,
            },
        }

    # No errors to repair
    if not validation_errors:
        logger.info("No validation errors to repair")
        return {
            **state,
            "needs_repair": False,
        }

    # No agent available
    if agent is None:
        logger.warning("No agent available for repair, cannot fix validation errors")
        return {
            **state,
            "needs_repair": False,
            "validated": False,
            "error": {
                "error_type": "validation",
                "message": "No agent available for repair",
                "node": "repair_code",
                "task_id": task_id,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    # Repair each file with validation errors
    repaired_code: dict[str, str] = {}
    repair_results: dict[str, dict] = {}

    for file_path, error_info in validation_errors.items():
        original_code = generated_code.get(file_path, "")

        if not original_code:
            logger.warning(f"No original code for {file_path}, skipping repair")
            continue

        error_type = error_info.get("error_type", "syntax")
        error_message = error_info.get("error_message") or error_info.get("error", "unknown error")
        error_line = error_info.get("error_line") or error_info.get("line", "unknown")

        # Build repair prompt
        template = _get_repair_prompt_template(error_type)
        prompt = template.format(
            error_type=error_type,
            line=error_line,
            message=error_message,
            code=original_code,
        )

        logger.info(
            f"Repairing {file_path} (attempt {repair_count + 1}/{MAX_REPAIRS}) - "
            f"{error_type} error at line {error_line}"
        )

        try:
            # Use agent to generate repaired code
            result = await agent.arun(prompt)
            fixed_code = _clean_repaired_code(str(result))

            repaired_code[file_path] = fixed_code
            repair_results[file_path] = {
                "status": "repaired",
                "original_error": error_message,
            }
            logger.info(f"Successfully generated repair for {file_path}")

        except Exception as e:
            logger.error(f"Failed to repair {file_path}: {e}")
            repair_results[file_path] = {
                "status": "failed",
                "error": str(e),
            }
            # Keep original code on failure
            repaired_code[file_path] = original_code

    # Merge repaired code with original
    updated_code = {**generated_code, **repaired_code}

    return {
        **state,
        "generated_code": updated_code,
        "repair_count": repair_count + 1,
        "needs_repair": False,  # Will be set again if validation fails
        "validation_errors": {},  # Clear for next validation cycle
        "repair_results": repair_results,  # Track what was repaired
    }


def should_repair(state: ExecutorGraphState) -> bool:
    """Check if repair should be attempted.

    Args:
        state: Current graph state.

    Returns:
        True if repair should be attempted.
    """
    needs_repair = state.get("needs_repair", False)
    repair_count = state.get("repair_count", 0)

    return needs_repair and repair_count < MAX_REPAIRS
