"""Verify task node for executor graph.

Phase 1.2.2: Verifies task completion using TaskVerifier.
Phase 1.4.3: Adapts TaskVerifier for graph (unchanged API).
Phase 3.2: Integrates VerificationAgent for autonomous verification.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.agents.verify_agent import (
    VerificationAgent,
    is_docs_only_change,
    task_needs_deep_verification,
)
from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.verifier import CheckLevel, TaskVerifier

logger = get_logger("executor.nodes.verify")

# Default timeout for autonomous verification (5 minutes)
DEFAULT_VERIFY_TIMEOUT: float = 300.0


async def verify_task_node(
    state: ExecutorGraphState,
    *,
    verifier: TaskVerifier | None = None,
    check_level: CheckLevel | None = None,
    workspace: Path | str | None = None,
) -> ExecutorGraphState:
    """Verify that the current task was completed successfully.

    Phase 1.4.3: TaskVerifier integration with graph (unchanged API).
    Phase 3.2: Integrates VerificationAgent for autonomous verification.

    Verification is performed in two levels:
    1. Level 1 (Fast): Syntax check using TaskVerifier (always, <1s)
    2. Level 2 (Autonomous): VerificationAgent runs tests (if enabled)

    The verifier runs checks, and results are stored in graph state
    for routing decisions:
    - verified=True -> route to checkpoint
    - verified=False + error -> route to handle_failure

    Args:
        state: Current graph state with current_task and files_modified.
        verifier: The TaskVerifier instance (Phase 1.4.3).
        check_level: Optional verification level (defaults to SYNTAX).
        workspace: Workspace path for autonomous verification.

    Returns:
        Updated state with verified flag or error.
    """
    current_task = state.get("current_task")
    files_modified = state.get("files_modified", [])
    task_id = str(current_task.id) if current_task else "unknown"

    if current_task is None:
        logger.error("No current task to verify")
        return {
            **state,
            "verified": False,
            "error": {
                "error_type": "verification",
                "message": "No current task to verify",
                "node": "verify_task",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    logger.info(f"Verifying task [{task_id}] with {len(files_modified)} files")

    # -------------------------------------------------------------------------
    # Level 1: Fast syntax check (always runs)
    # -------------------------------------------------------------------------
    syntax_result = await _check_syntax(state, verifier, check_level)
    if not syntax_result["passed"]:
        logger.warning(f"Task [{task_id}] failed syntax check: {syntax_result['error']}")
        return {
            **state,
            "verified": False,
            "error": {
                "error_type": "verification",
                "message": syntax_result["error"],
                "node": "verify_task",
                "task_id": task_id,
                "recoverable": True,
                "stack_trace": None,
            },
        }

    # -------------------------------------------------------------------------
    # Level 2: Autonomous verification (if enabled)
    # -------------------------------------------------------------------------
    enable_autonomous = state.get("enable_autonomous_verify", False)

    if enable_autonomous:
        # Check if task needs deep verification using heuristics
        if not _should_run_autonomous_verify(current_task, files_modified):
            logger.info(
                f"Task [{task_id}] skipped autonomous verification (docs-only or simple change)"
            )
            return {
                **state,
                "verified": True,
                "error": None,
                "autonomous_verify_result": {
                    "passed": True,
                    "checks_run": ["skipped-heuristic"],
                    "failures": [],
                    "suggestions": [],
                    "duration_ms": 0.0,
                },
            }

        # Run autonomous verification
        autonomous_result = await _run_autonomous_verification(
            state=state,
            current_task=current_task,
            files_modified=files_modified,
            workspace=workspace,
        )

        if not autonomous_result["passed"]:
            failure_summary = _format_verification_failures(autonomous_result)
            logger.warning(f"Task [{task_id}] failed autonomous verification: {failure_summary}")
            return {
                **state,
                "verified": False,
                "error": {
                    "error_type": "verification",
                    "message": failure_summary,
                    "node": "verify_task",
                    "task_id": task_id,
                    "recoverable": True,
                    "stack_trace": None,
                },
                "autonomous_verify_result": autonomous_result,
            }

        logger.info(f"Task [{task_id}] passed autonomous verification")
        return {
            **state,
            "verified": True,
            "error": None,
            "autonomous_verify_result": autonomous_result,
        }

    # No autonomous verification - just syntax check passed
    logger.info(f"Task [{task_id}] verified successfully (syntax only)")
    return {
        **state,
        "verified": True,
        "error": None,
    }


async def _check_syntax(
    state: ExecutorGraphState,
    verifier: TaskVerifier | None,
    check_level: CheckLevel | None,
) -> dict[str, Any]:
    """Perform fast syntax check using TaskVerifier.

    Phase 3.2.1: Level 1 verification (always runs, <1s).

    Args:
        state: Current graph state.
        verifier: The TaskVerifier instance.
        check_level: Verification level (defaults to SYNTAX).

    Returns:
        Dict with 'passed' (bool) and 'error' (str or None).
    """
    current_task = state.get("current_task")

    # If no verifier, skip syntax check (development mode)
    if verifier is None:
        logger.warning("No verifier provided, skipping syntax check")
        return {"passed": True, "error": None}

    try:
        # Get default check level if not specified
        if check_level is None:
            from ai_infra.executor.verifier import CheckLevel

            check_level = CheckLevel.SYNTAX

        # Run verification with timeout
        verification_result = await asyncio.wait_for(
            verifier.verify(
                task=current_task,
                levels=[check_level],
            ),
            timeout=NodeTimeouts.VERIFY_TASK,
        )

        if verification_result.overall:
            return {"passed": True, "error": None}
        else:
            failures = verification_result.get_failures()
            failure_message = failures[0].message if failures else verification_result.summary()
            return {"passed": False, "error": failure_message}

    except TimeoutError:
        return {
            "passed": False,
            "error": f"Syntax check timed out after {NodeTimeouts.VERIFY_TASK}s",
        }

    except Exception as e:
        logger.exception(f"Syntax check error: {e}")
        return {"passed": False, "error": str(e)}


def _should_run_autonomous_verify(task: Any, files_modified: list[str]) -> bool:
    """Determine if autonomous verification should run.

    Phase 3.2.2: Heuristic to decide if deep verification is needed.

    Args:
        task: The current task.
        files_modified: List of modified file paths.

    Returns:
        True if autonomous verification should run.
    """
    # Skip for docs-only changes
    if is_docs_only_change(files_modified):
        return False

    # Run for tasks that need deep verification
    if task_needs_deep_verification(task):
        return True

    # Run if any code files were modified
    code_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".rb"}
    for file_path in files_modified:
        if any(file_path.endswith(ext) for ext in code_extensions):
            return True

    return False


async def _run_autonomous_verification(
    state: ExecutorGraphState,
    current_task: Any,
    files_modified: list[str],
    workspace: Path | str | None,
) -> dict[str, Any]:
    """Run autonomous verification using VerificationAgent.

    Phase 3.2.1 & 3.2.3: Runs the verification agent with timeout handling.

    Args:
        state: Current graph state.
        current_task: The task to verify.
        files_modified: List of modified files.
        workspace: Workspace path.

    Returns:
        Dict with verification results (passed, checks_run, failures, etc.).
    """
    # Get workspace path
    if workspace is None:
        roadmap_path = state.get("roadmap_path")
        if roadmap_path:
            workspace = Path(roadmap_path).parent
        else:
            workspace = Path.cwd()
    elif isinstance(workspace, str):
        workspace = Path(workspace)

    # Get timeout from state or use default (Phase 3.2.3: 5 minutes max)
    verify_timeout = state.get("verify_timeout", DEFAULT_VERIFY_TIMEOUT)

    # Create verification agent
    agent = VerificationAgent(
        timeout=verify_timeout,
        skip_docs_only=False,  # Already checked in _should_run_autonomous_verify
    )

    try:
        # Run with timeout (Phase 3.2.3)
        result = await asyncio.wait_for(
            agent.verify(
                workspace=workspace,
                task=current_task,
                files_modified=files_modified,
            ),
            timeout=verify_timeout,
        )

        return result.to_dict()

    except TimeoutError:
        logger.warning(f"Autonomous verification timed out after {verify_timeout}s")
        # Return partial results on timeout (Phase 3.2.3)
        return {
            "passed": False,
            "checks_run": [],
            "failures": [
                {
                    "command": "autonomous_verification",
                    "exit_code": -1,
                    "error": f"Verification timed out after {verify_timeout}s",
                }
            ],
            "suggestions": ["Consider increasing --verify-timeout or simplifying tests."],
            "duration_ms": verify_timeout * 1000,
        }

    except Exception as e:
        logger.exception(f"Autonomous verification error: {e}")
        return {
            "passed": False,
            "checks_run": [],
            "failures": [
                {
                    "command": "autonomous_verification",
                    "exit_code": -1,
                    "error": str(e),
                }
            ],
            "suggestions": [],
            "duration_ms": 0.0,
        }


def _format_verification_failures(result: dict[str, Any]) -> str:
    """Format verification failures into a human-readable message.

    Args:
        result: Verification result dict.

    Returns:
        Formatted failure message.
    """
    failures = result.get("failures", [])
    if not failures:
        return "Verification failed (no details available)"

    messages = []
    for failure in failures[:3]:  # Limit to first 3 failures
        error = failure.get("error", "Unknown error")
        command = failure.get("command", "")
        if command:
            messages.append(f"[{command}] {error}")
        else:
            messages.append(error)

    return "; ".join(messages)
