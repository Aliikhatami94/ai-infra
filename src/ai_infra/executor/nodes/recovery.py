"""Recovery node for intelligent task recovery.

Phase 2.9.1: Failure classification and recovery proposal types.
Phase 2.9: Intelligent task recovery for enterprise use cases.

When a task fails after max repairs, this module provides:
- Failure classification (why did it fail?)
- Recovery strategy selection (how to recover?)
- Recovery proposal generation (what changes to make?)

See EXECUTOR.md Phase 2.9 for full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent
    from ai_infra.executor.todolist import TodoItem

logger = get_logger("executor.nodes.recovery")


# =============================================================================
# Phase 2.9.1: Failure Classification
# =============================================================================


class FailureReason(str, Enum):
    """Root cause classification for task failures.

    Used to determine the appropriate recovery strategy when a task
    fails after max repair attempts.

    Attributes:
        TASK_TOO_VAGUE: Task description is unclear or ambiguous.
        TASK_TOO_COMPLEX: Task is trying to do too many things at once.
        MISSING_DEPENDENCY: Task requires something that does not exist yet.
        ENVIRONMENT_ISSUE: Missing package, wrong version, system config problem.
        UNKNOWN: Cannot determine the root cause.
    """

    TASK_TOO_VAGUE = "task_too_vague"
    """Task description is unclear or ambiguous - needs rewriting."""

    TASK_TOO_COMPLEX = "task_too_complex"
    """Task is trying to do too many things at once - needs decomposition."""

    MISSING_DEPENDENCY = "missing_dependency"
    """Task requires something that does not exist yet - needs deferral."""

    ENVIRONMENT_ISSUE = "environment_issue"
    """Missing package, wrong Python version, or system config problem."""

    UNKNOWN = "unknown"
    """Cannot determine the root cause - requires human escalation."""


class RecoveryStrategy(str, Enum):
    """Strategy for recovering from a task failure.

    Each strategy corresponds to a specific action the executor will take
    to attempt recovery.

    Attributes:
        REWRITE: Clarify the task description with specific requirements.
        DECOMPOSE: Split the task into 2-5 smaller, independent tasks.
        DEFER: Retry the task after other tasks complete.
        SKIP: Skip the task and log an error (human must fix environment).
        ESCALATE: Pause execution and notify human for review.
    """

    REWRITE = "rewrite"
    """Clarify the task description with specific requirements."""

    DECOMPOSE = "decompose"
    """Split the task into 2-5 smaller, independent tasks."""

    DEFER = "defer"
    """Retry the task after other tasks complete."""

    SKIP = "skip"
    """Skip the task and log an error (human must fix environment)."""

    ESCALATE = "escalate"
    """Pause execution and notify human for review."""


# Mapping from failure reason to recovery strategy
FAILURE_TO_STRATEGY: dict[FailureReason, RecoveryStrategy] = {
    FailureReason.TASK_TOO_VAGUE: RecoveryStrategy.REWRITE,
    FailureReason.TASK_TOO_COMPLEX: RecoveryStrategy.DECOMPOSE,
    FailureReason.MISSING_DEPENDENCY: RecoveryStrategy.DEFER,
    FailureReason.ENVIRONMENT_ISSUE: RecoveryStrategy.SKIP,
    FailureReason.UNKNOWN: RecoveryStrategy.ESCALATE,
}
"""Maps each failure reason to its corresponding recovery strategy."""


@dataclass
class RecoveryProposal:
    """Proposed recovery action for human review.

    Contains all information needed for a human to understand why a task
    failed and approve/reject the proposed recovery action.

    Attributes:
        original_task: The task that failed.
        failure_reason: Why the task failed.
        strategy: The proposed recovery strategy.
        proposed_tasks: List of rewritten or decomposed task descriptions.
        explanation: Human-readable explanation of why this recovery was chosen.
        requires_approval: Whether human approval is required before applying.
        error_history: List of error messages from repair attempts.
    """

    original_task: str
    """The original task description that failed."""

    failure_reason: FailureReason
    """Classification of why the task failed."""

    strategy: RecoveryStrategy
    """The recovery strategy to apply."""

    proposed_tasks: list[str] = field(default_factory=list)
    """Rewritten or decomposed task descriptions."""

    explanation: str = ""
    """Human-readable explanation of why this recovery was chosen."""

    requires_approval: bool = True
    """Whether human approval is required before applying the recovery."""

    error_history: list[str] = field(default_factory=list)
    """List of error messages from repair attempts."""

    def to_markdown(self) -> str:
        """Format the proposal as Markdown for human review.

        Returns:
            A formatted Markdown string suitable for display or logging.
        """
        lines = [
            "## Recovery Proposal",
            "",
            f"**Original Task**: {self.original_task}",
            f"**Failure Reason**: {self.failure_reason.value}",
            f"**Strategy**: {self.strategy.value}",
            f"**Requires Approval**: {'Yes' if self.requires_approval else 'No'}",
            "",
        ]

        if self.explanation:
            lines.extend(
                [
                    "### Explanation",
                    "",
                    self.explanation,
                    "",
                ]
            )

        if self.proposed_tasks:
            lines.extend(
                [
                    "### Proposed Replacement",
                    "",
                ]
            )
            for task in self.proposed_tasks:
                lines.append(f"- [ ] {task}")
            lines.append("")

        if self.error_history:
            lines.extend(
                [
                    "### Error History",
                    "",
                ]
            )
            for i, error in enumerate(self.error_history, 1):
                lines.append(f"{i}. {error}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_task(
        cls,
        task: TodoItem,
        failure_reason: FailureReason,
        proposed_tasks: list[str] | None = None,
        explanation: str = "",
        error_history: list[str] | None = None,
    ) -> RecoveryProposal:
        """Create a RecoveryProposal from a TodoItem.

        Args:
            task: The TodoItem that failed.
            failure_reason: Why the task failed.
            proposed_tasks: List of replacement task descriptions.
            explanation: Human-readable explanation.
            error_history: List of error messages.

        Returns:
            A RecoveryProposal instance.
        """
        strategy = FAILURE_TO_STRATEGY[failure_reason]

        # Determine if approval is required based on strategy
        requires_approval = strategy in (
            RecoveryStrategy.DECOMPOSE,
            RecoveryStrategy.REWRITE,
            RecoveryStrategy.ESCALATE,
        )

        return cls(
            original_task=task.title,
            failure_reason=failure_reason,
            strategy=strategy,
            proposed_tasks=proposed_tasks or [],
            explanation=explanation,
            requires_approval=requires_approval,
            error_history=error_history or [],
        )


def get_strategy_for_failure(reason: FailureReason) -> RecoveryStrategy:
    """Get the recovery strategy for a given failure reason.

    Args:
        reason: The classified failure reason.

    Returns:
        The corresponding recovery strategy.

    Example:
        >>> get_strategy_for_failure(FailureReason.TASK_TOO_COMPLEX)
        <RecoveryStrategy.DECOMPOSE: 'decompose'>
    """
    return FAILURE_TO_STRATEGY[reason]


# =============================================================================
# Phase 2.9.2: Classify Failure Node
# =============================================================================

# Pattern lists for fast classification (no LLM needed)
ENVIRONMENT_ISSUE_PATTERNS = [
    "modulenotfounderror",
    "no module named",
    "pip install",
    "permission denied",
    "access denied",
    "command not found",
    "executable not found",
    "python version",
    "unsupported python",
    "missing dependency",
    "package not found",
    "could not find a version",
    "incompatible",
]

MISSING_DEPENDENCY_PATTERNS = [
    "importerror",
    "cannot import",
    "no such file",
    "file not found",
    "does not exist",
    "undefined reference",
    "unresolved import",
    "circular import",
]

TASK_TOO_VAGUE_PATTERNS = [
    "unclear",
    "ambiguous",
    "not specified",
    "missing information",
    "need more details",
]

CLASSIFY_FAILURE_PROMPT = """Analyze why this task failed after multiple repair attempts.

## Task Description
{task_description}

## Error History
{error_history}

## Last Generated Code
```
{last_code}
```

## Last Error
{last_error}

## Classification Options
1. task_too_vague - The task description is unclear or ambiguous
2. task_too_complex - The task is trying to do too many things at once
3. missing_dependency - The task requires something that doesn't exist yet
4. environment_issue - Missing package, wrong version, system config problem
5. unknown - Cannot determine the root cause

Respond with ONLY the classification (e.g., "task_too_complex"), nothing else.
"""


def _classify_by_pattern(
    error_message: str,
    state: dict[str, Any],
) -> FailureReason:
    """Fast pattern-based classification without LLM.

    Examines error messages and task characteristics to classify
    the failure reason using simple pattern matching.

    Args:
        error_message: The error message from the last failure.
        state: The executor graph state.

    Returns:
        The classified FailureReason, or UNKNOWN if no pattern matches.
    """
    msg = error_message.lower()

    # Check for environment issues (most specific first)
    if any(pattern in msg for pattern in ENVIRONMENT_ISSUE_PATTERNS):
        return FailureReason.ENVIRONMENT_ISSUE

    # Check for missing dependency patterns
    if any(pattern in msg for pattern in MISSING_DEPENDENCY_PATTERNS):
        return FailureReason.MISSING_DEPENDENCY

    # Check for vague task patterns
    if any(pattern in msg for pattern in TASK_TOO_VAGUE_PATTERNS):
        return FailureReason.TASK_TOO_VAGUE

    # Check task complexity heuristics
    current_task = state.get("current_task")
    if current_task is not None:
        # Get description - handle both dict and TodoItem
        if hasattr(current_task, "description"):
            task_desc = current_task.description or ""
        elif isinstance(current_task, dict):
            task_desc = current_task.get("description", "")
        else:
            task_desc = ""

        # Heuristics for complexity
        if len(task_desc) > 500:
            return FailureReason.TASK_TOO_COMPLEX
        if task_desc.count(",") > 5:
            return FailureReason.TASK_TOO_COMPLEX
        if task_desc.count(" and ") > 3:
            return FailureReason.TASK_TOO_COMPLEX

    return FailureReason.UNKNOWN


def _format_error_history(state: dict[str, Any]) -> str:
    """Format error history from state for the classification prompt.

    Args:
        state: The executor graph state.

    Returns:
        Formatted string of error history.
    """
    error_history = state.get("error_history", [])
    if not error_history:
        error = state.get("error", {})
        if error:
            return f"1. {error.get('message', 'Unknown error')}"
        return "No error history available."

    lines = []
    for i, error in enumerate(error_history, 1):
        if isinstance(error, dict):
            msg = error.get("message", str(error))
        else:
            msg = str(error)
        lines.append(f"{i}. {msg}")

    return "\n".join(lines)


def _get_last_code(state: dict[str, Any]) -> str:
    """Extract the last generated code from state.

    Args:
        state: The executor graph state.

    Returns:
        The last generated code, or a placeholder if not available.
    """
    generated_code = state.get("generated_code", {})
    if not generated_code:
        return "(No code available)"

    # Return the first file's content as sample
    for file_path, content in generated_code.items():
        if isinstance(content, str):
            # Truncate to avoid huge prompts
            if len(content) > 2000:
                return content[:2000] + "\n... (truncated)"
            return content

    return "(No code available)"


def _parse_classification(response: str) -> FailureReason:
    """Parse LLM response into a FailureReason.

    Args:
        response: The raw LLM response.

    Returns:
        The parsed FailureReason, or UNKNOWN if parsing fails.
    """
    response = response.strip().lower()

    # Direct match
    for reason in FailureReason:
        if reason.value == response:
            return reason

    # Partial match (in case LLM includes extra text)
    for reason in FailureReason:
        if reason.value in response:
            return reason

    logger.warning(f"Could not parse classification from response: {response}")
    return FailureReason.UNKNOWN


async def classify_failure_node(
    state: dict[str, Any],
    *,
    agent: Agent | None = None,
) -> dict[str, Any]:
    """Classify why a task failed to determine recovery strategy.

    Uses pattern matching first (fast, no cost), then falls back to
    LLM classification if pattern matching returns UNKNOWN.

    Args:
        state: The executor graph state.
        agent: Optional agent for LLM-based classification.

    Returns:
        Updated state with failure_reason and recovery_strategy fields.

    Example:
        >>> state = {"error": {"message": "ModuleNotFoundError: No module named 'foo'"}}
        >>> result = await classify_failure_node(state)
        >>> result["failure_reason"]
        'environment_issue'
        >>> result["recovery_strategy"]
        'skip'
    """
    error = state.get("error", {})
    error_message = error.get("message", "") if isinstance(error, dict) else str(error)

    # Try fast pattern-based classification first
    reason = _classify_by_pattern(error_message, state)

    # Fall back to LLM if pattern matching didn't work
    if reason == FailureReason.UNKNOWN and agent is not None:
        current_task = state.get("current_task")
        if hasattr(current_task, "description"):
            task_desc = current_task.description or ""
        elif isinstance(current_task, dict):
            task_desc = current_task.get("description", "")
        else:
            task_desc = str(current_task) if current_task else ""

        prompt = CLASSIFY_FAILURE_PROMPT.format(
            task_description=task_desc,
            error_history=_format_error_history(state),
            last_code=_get_last_code(state),
            last_error=error_message,
        )

        try:
            response = await agent.arun(prompt)
            reason = _parse_classification(str(response))
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            reason = FailureReason.UNKNOWN

    strategy = FAILURE_TO_STRATEGY[reason]

    logger.info(
        "failure_classified",
        reason=reason.value,
        strategy=strategy.value,
        error_message=error_message[:100] if error_message else None,
    )

    return {
        **state,
        "failure_reason": reason.value,
        "recovery_strategy": strategy.value,
    }


# =============================================================================
# Phase 2.9.3: Propose Recovery Node
# =============================================================================

REWRITE_TASK_PROMPT = """The following task failed because it was too vague.
Rewrite it to be specific and actionable.

## Original Task
{original_task}

## Failure Details
{failure_details}

## Project Context
{project_context}

## Instructions
1. Make the task specific (exact file paths, function names, behaviors)
2. Include clear success criteria
3. Keep scope similar to original intent
4. Return ONLY the rewritten task text, nothing else

## Rewritten Task
"""

DECOMPOSE_TASK_PROMPT = """The following task failed because it was too complex.
Break it into 2-5 smaller, independent tasks.

## Original Task
{original_task}

## Failure Details
{failure_details}

## Project Context
{project_context}

## Instructions
1. Each sub-task should be achievable in a single file
2. Order tasks by dependency (what must be done first)
3. Each task should have clear, testable output
4. Return ONLY a numbered list of tasks, one per line

## Sub-Tasks
"""


def _format_failure_details(state: dict[str, Any]) -> str:
    """Format failure details for recovery prompts.

    Args:
        state: The executor graph state.

    Returns:
        Formatted string with failure details.
    """
    lines = []

    # Error message
    error = state.get("error", {})
    if isinstance(error, dict):
        error_msg = error.get("message", "Unknown error")
    else:
        error_msg = str(error) if error else "Unknown error"
    lines.append(f"**Error**: {error_msg}")

    # Repair count
    repair_count = state.get("repair_count", 0)
    test_repair_count = state.get("test_repair_count", 0)
    if repair_count > 0:
        lines.append(f"**Validation Repairs Attempted**: {repair_count}")
    if test_repair_count > 0:
        lines.append(f"**Test Repairs Attempted**: {test_repair_count}")

    # Error history summary
    error_history = state.get("error_history", [])
    if error_history:
        lines.append(f"**Previous Errors**: {len(error_history)} failures before this")

    return "\n".join(lines)


def _get_project_summary(state: dict[str, Any]) -> str:
    """Get a brief project summary for context.

    Args:
        state: The executor graph state.

    Returns:
        Project context summary.
    """
    lines = []

    # Roadmap path
    roadmap_path = state.get("roadmap_path", "")
    if roadmap_path:
        lines.append(f"**Roadmap**: {roadmap_path}")

    # Todo list summary
    todo_list = state.get("todo_list", [])
    if todo_list:
        completed = sum(1 for t in todo_list if _get_task_status(t) == "completed")
        total = len(todo_list)
        lines.append(f"**Progress**: {completed}/{total} tasks completed")

    # Current task position
    current_task = state.get("current_task")
    if current_task:
        task_id = _get_task_id(current_task)
        if task_id:
            lines.append(f"**Current Task ID**: {task_id}")

    if not lines:
        return "(No project context available)"

    return "\n".join(lines)


def _get_task_status(task: Any) -> str:
    """Get status from a task (dict or TodoItem)."""
    if hasattr(task, "status"):
        status = task.status
        return status.value if hasattr(status, "value") else str(status)
    elif isinstance(task, dict):
        return task.get("status", "unknown")
    return "unknown"


def _get_task_id(task: Any) -> str | None:
    """Get ID from a task (dict or TodoItem)."""
    if hasattr(task, "id"):
        return str(task.id)
    elif isinstance(task, dict):
        return str(task.get("id", ""))
    return None


def _get_task_title(task: Any) -> str:
    """Get title from a task (dict or TodoItem)."""
    if hasattr(task, "title"):
        return task.title or ""
    elif isinstance(task, dict):
        return task.get("title", "")
    return ""


def _get_task_description(task: Any) -> str:
    """Get description from a task (dict or TodoItem)."""
    if hasattr(task, "description"):
        return task.description or ""
    elif isinstance(task, dict):
        return task.get("description", "")
    return ""


def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list from LLM response.

    Handles formats like:
    - "1. First task"
    - "1) First task"
    - "- First task"
    - "* First task"

    Args:
        text: The raw LLM response.

    Returns:
        List of task descriptions.
    """
    import re

    lines = text.strip().split("\n")
    tasks = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common list prefixes
        # Match: "1.", "1)", "-", "*", "•"
        cleaned = re.sub(r"^(\d+[.\)]\s*|[-*•]\s*)", "", line)
        cleaned = cleaned.strip()

        if cleaned:
            tasks.append(cleaned)

    return tasks


async def propose_recovery_node(
    state: dict[str, Any],
    *,
    agent: Agent | None = None,
) -> dict[str, Any]:
    """Generate a recovery proposal based on the classified failure.

    Creates a RecoveryProposal with strategy-specific handling:
    - REWRITE: Uses LLM to clarify vague task description
    - DECOMPOSE: Uses LLM to split complex task into sub-tasks
    - DEFER: Marks task to retry after dependencies complete
    - SKIP: Marks task as skipped (environment issue)
    - ESCALATE: Pauses for human review

    Args:
        state: The executor graph state with failure_reason and recovery_strategy.
        agent: Optional agent for LLM-based proposal generation.

    Returns:
        Updated state with recovery_proposal field.

    Example:
        >>> state = {
        ...     "recovery_strategy": "decompose",
        ...     "failure_reason": "task_too_complex",
        ...     "current_task": {"title": "Build entire system", "description": "..."},
        ... }
        >>> result = await propose_recovery_node(state, agent=mock_agent)
        >>> result["recovery_proposal"].strategy
        <RecoveryStrategy.DECOMPOSE: 'decompose'>
    """
    strategy_value = state.get("recovery_strategy", "escalate")
    failure_reason_value = state.get("failure_reason", "unknown")

    try:
        strategy = RecoveryStrategy(strategy_value)
    except ValueError:
        strategy = RecoveryStrategy.ESCALATE

    try:
        failure_reason = FailureReason(failure_reason_value)
    except ValueError:
        failure_reason = FailureReason.UNKNOWN

    current_task = state.get("current_task")
    original_task = _get_task_description(current_task) or _get_task_title(current_task)
    task_title = _get_task_title(current_task)

    proposal: RecoveryProposal | None = None

    if strategy == RecoveryStrategy.REWRITE:
        if agent is not None:
            prompt = REWRITE_TASK_PROMPT.format(
                original_task=original_task,
                failure_details=_format_failure_details(state),
                project_context=_get_project_summary(state),
            )
            try:
                rewritten = await agent.arun(prompt)
                proposal = RecoveryProposal(
                    original_task=task_title or original_task,
                    failure_reason=failure_reason,
                    strategy=strategy,
                    proposed_tasks=[str(rewritten).strip()],
                    explanation="Task was unclear. Rewritten with specific requirements.",
                    requires_approval=True,
                    error_history=_collect_error_messages(state),
                )
            except Exception as e:
                logger.warning(f"LLM rewrite failed: {e}")
                # Fall back to escalate
                proposal = _create_escalate_proposal(
                    task_title or original_task,
                    failure_reason,
                    f"Rewrite failed: {e}",
                    state,
                )
        else:
            # No agent, cannot rewrite - escalate instead
            proposal = _create_escalate_proposal(
                task_title or original_task,
                failure_reason,
                "No agent available for rewrite. Human intervention required.",
                state,
            )

    elif strategy == RecoveryStrategy.DECOMPOSE:
        if agent is not None:
            prompt = DECOMPOSE_TASK_PROMPT.format(
                original_task=original_task,
                failure_details=_format_failure_details(state),
                project_context=_get_project_summary(state),
            )
            try:
                decomposed = await agent.arun(prompt)
                sub_tasks = _parse_numbered_list(str(decomposed))
                if sub_tasks:
                    proposal = RecoveryProposal(
                        original_task=task_title or original_task,
                        failure_reason=failure_reason,
                        strategy=strategy,
                        proposed_tasks=sub_tasks,
                        explanation=f"Task was too complex. Decomposed into {len(sub_tasks)} smaller tasks.",
                        requires_approval=True,
                        error_history=_collect_error_messages(state),
                    )
                else:
                    # Decompose returned nothing useful
                    proposal = _create_escalate_proposal(
                        task_title or original_task,
                        failure_reason,
                        "Decomposition produced no valid sub-tasks.",
                        state,
                    )
            except Exception as e:
                logger.warning(f"LLM decompose failed: {e}")
                proposal = _create_escalate_proposal(
                    task_title or original_task,
                    failure_reason,
                    f"Decomposition failed: {e}",
                    state,
                )
        else:
            proposal = _create_escalate_proposal(
                task_title or original_task,
                failure_reason,
                "No agent available for decomposition. Human intervention required.",
                state,
            )

    elif strategy == RecoveryStrategy.DEFER:
        proposal = RecoveryProposal(
            original_task=task_title or original_task,
            failure_reason=failure_reason,
            strategy=strategy,
            proposed_tasks=[original_task] if original_task else [],
            explanation="Task depends on something not yet implemented. Will retry after other tasks complete.",
            requires_approval=False,
            error_history=_collect_error_messages(state),
        )

    elif strategy == RecoveryStrategy.SKIP:
        proposal = RecoveryProposal(
            original_task=task_title or original_task,
            failure_reason=failure_reason,
            strategy=strategy,
            proposed_tasks=[],
            explanation="Environment issue detected. Task skipped. Human intervention required to fix environment.",
            requires_approval=False,
            error_history=_collect_error_messages(state),
        )

    elif strategy == RecoveryStrategy.ESCALATE:
        proposal = _create_escalate_proposal(
            task_title or original_task,
            failure_reason,
            "Cannot determine automatic recovery path. Pausing for human review.",
            state,
        )

    # Log the proposal
    if proposal is not None:
        logger.info(
            "recovery_proposed",
            strategy=proposal.strategy.value,
            failure_reason=proposal.failure_reason.value,
            proposed_tasks_count=len(proposal.proposed_tasks),
            requires_approval=proposal.requires_approval,
        )

    return {
        **state,
        "recovery_proposal": proposal,
    }


def _create_escalate_proposal(
    original_task: str,
    failure_reason: FailureReason,
    explanation: str,
    state: dict[str, Any],
) -> RecoveryProposal:
    """Create a fallback escalate proposal.

    Args:
        original_task: The original task description.
        failure_reason: The classified failure reason.
        explanation: Why escalation is needed.
        state: The executor graph state.

    Returns:
        A RecoveryProposal with ESCALATE strategy.
    """
    return RecoveryProposal(
        original_task=original_task,
        failure_reason=failure_reason,
        strategy=RecoveryStrategy.ESCALATE,
        proposed_tasks=[],
        explanation=explanation,
        requires_approval=True,
        error_history=_collect_error_messages(state),
    )


def _collect_error_messages(state: dict[str, Any]) -> list[str]:
    """Collect error messages from state for the proposal.

    Args:
        state: The executor graph state.

    Returns:
        List of error message strings.
    """
    messages = []

    # Current error
    error = state.get("error", {})
    if isinstance(error, dict):
        msg = error.get("message")
        if msg:
            messages.append(str(msg))
    elif error:
        messages.append(str(error))

    # Error history
    error_history = state.get("error_history", [])
    for err in error_history:
        if isinstance(err, dict):
            msg = err.get("message")
            if msg:
                messages.append(str(msg))
        elif err:
            messages.append(str(err))

    return messages


# =============================================================================
# Phase 2.9.4: Approval Modes (Enterprise Control)
# =============================================================================


class ApprovalMode(str, Enum):
    """How to handle recovery proposals that modify the roadmap.

    Controls whether the executor pauses for human approval before
    applying recovery actions.

    Attributes:
        AUTO: Apply all recoveries automatically (for CI/CD pipelines).
        INTERACTIVE: Pause and ask for each recovery (careful mode).
        REVIEW_ONLY: Log proposals but don't apply them (dry-run).
        APPROVE_DECOMPOSE: Auto-approve rewrites, ask only for decompose.
    """

    AUTO = "auto"
    """Apply all recoveries automatically without human intervention."""

    INTERACTIVE = "interactive"
    """Pause and prompt user for approval on each recovery."""

    REVIEW_ONLY = "review_only"
    """Log proposals but don't apply them (dry-run mode)."""

    APPROVE_DECOMPOSE = "approve_decompose"
    """Auto-approve rewrites, but pause and ask for decompose operations."""


def _interactive_edit(
    tasks: list[str],
    get_input: callable | None = None,
) -> list[str]:
    """Allow user to interactively edit proposed tasks.

    Presents tasks in a simple editor format where users can modify,
    add, or remove tasks.

    Args:
        tasks: The current list of proposed tasks.
        get_input: Optional callable for testing (replaces input()).

    Returns:
        The edited list of tasks.
    """
    _input = get_input if get_input is not None else input

    print("\n--- Edit Proposed Tasks ---")
    print("Enter tasks one per line. Empty line to finish.")
    print("Current tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    print("\nEnter new tasks (or press Enter to keep current):")

    new_tasks = []
    while True:
        try:
            line = _input(f"  {len(new_tasks) + 1}. ").strip()
            if not line:
                break
            new_tasks.append(line)
        except (EOFError, KeyboardInterrupt):
            break

    # If no new tasks entered, keep original
    if not new_tasks:
        return tasks

    return new_tasks


async def await_approval_node(
    state: dict[str, Any],
    *,
    approval_mode: ApprovalMode | str | None = None,
    interactive_input: callable | None = None,
) -> dict[str, Any]:
    """Wait for human approval if required based on approval mode.

    Behavior depends on the approval_mode:
    - AUTO: Skip approval, automatically approve all recoveries
    - INTERACTIVE: Pause and prompt user for approval
    - REVIEW_ONLY: Log the proposal but don't apply (dry-run)
    - APPROVE_DECOMPOSE: Auto-approve rewrites, ask for decompose

    Args:
        state: The executor graph state with recovery_proposal.
        approval_mode: Override the approval mode from state.
        interactive_input: Optional callable for testing (replaces input()).

    Returns:
        Updated state with recovery_approved and optional recovery_skipped.

    Example:
        >>> state = {"recovery_proposal": proposal, "approval_mode": "auto"}
        >>> result = await await_approval_node(state)
        >>> result["recovery_approved"]
        True
    """
    proposal: RecoveryProposal | None = state.get("recovery_proposal")

    if proposal is None:
        return {**state, "recovery_approved": False}

    # Determine approval mode
    if approval_mode is not None:
        if isinstance(approval_mode, str):
            try:
                mode = ApprovalMode(approval_mode)
            except ValueError:
                mode = ApprovalMode.AUTO
        else:
            mode = approval_mode
    else:
        mode_value = state.get("approval_mode", "auto")
        try:
            mode = ApprovalMode(mode_value)
        except ValueError:
            mode = ApprovalMode.AUTO

    # Use provided input function or built-in input
    get_input = interactive_input if interactive_input is not None else input

    # Handle REVIEW_ONLY mode first - just log and skip
    if mode == ApprovalMode.REVIEW_ONLY:
        logger.info(
            "review_only_mode",
            reason="Recovery proposal logged but not applied",
            strategy=proposal.strategy.value,
            original_task=proposal.original_task,
        )
        logger.info(proposal.to_markdown())
        return {
            **state,
            "recovery_approved": False,
            "recovery_skipped": True,
        }

    # Determine if we need approval based on mode and strategy
    needs_approval = False

    if mode == ApprovalMode.INTERACTIVE:
        # Interactive mode always asks
        needs_approval = True
    elif mode == ApprovalMode.APPROVE_DECOMPOSE:
        # Only ask for DECOMPOSE operations
        needs_approval = proposal.strategy == RecoveryStrategy.DECOMPOSE
    elif proposal.strategy == RecoveryStrategy.ESCALATE:
        # ESCALATE always requires approval regardless of mode
        needs_approval = True
    # AUTO mode: needs_approval stays False

    if not needs_approval:
        logger.info(
            "recovery_auto_approved",
            strategy=proposal.strategy.value,
            mode=mode.value,
        )
        return {**state, "recovery_approved": True}

    # Interactive approval flow
    _print_approval_header(proposal)

    while True:
        try:
            response = (
                get_input("\nApprove this recovery? [y]es / [n]o / [e]dit / [s]kip task: ")
                .lower()
                .strip()
            )
        except (EOFError, KeyboardInterrupt):
            logger.info("recovery_cancelled", reason="User cancelled approval")
            return {**state, "recovery_approved": False, "task_skipped": True}

        if response in ("y", "yes"):
            logger.info("recovery_approved", by="user")
            return {**state, "recovery_approved": True}

        elif response in ("n", "no"):
            logger.info("recovery_rejected", by="user")
            return {**state, "recovery_approved": False}

        elif response in ("e", "edit"):
            # Let user edit the proposed tasks
            edited_tasks = _interactive_edit(proposal.proposed_tasks, get_input)
            # Create updated proposal with edited tasks
            updated_proposal = RecoveryProposal(
                original_task=proposal.original_task,
                failure_reason=proposal.failure_reason,
                strategy=proposal.strategy,
                proposed_tasks=edited_tasks,
                explanation=proposal.explanation,
                requires_approval=proposal.requires_approval,
                error_history=proposal.error_history,
            )
            logger.info("recovery_edited", new_task_count=len(edited_tasks))
            return {
                **state,
                "recovery_proposal": updated_proposal,
                "recovery_approved": True,
            }

        elif response in ("s", "skip"):
            logger.info("task_skipped", by="user")
            return {
                **state,
                "recovery_approved": False,
                "task_skipped": True,
            }

        else:
            print("Invalid option. Please enter y, n, e, or s.")


def _print_approval_header(proposal: RecoveryProposal) -> None:
    """Print the approval request header.

    Args:
        proposal: The recovery proposal to display.
    """
    print("\n" + "=" * 60)
    print("RECOVERY PROPOSAL - Approval Required")
    print("=" * 60)
    print(proposal.to_markdown())
    print("=" * 60)


# =============================================================================
# Phase 2.9.5: Apply Recovery Node
# =============================================================================


async def apply_recovery_node(
    state: dict[str, Any],
    *,
    roadmap_reader: callable | None = None,
    roadmap_writer: callable | None = None,
) -> dict[str, Any]:
    """Apply the approved recovery by updating ROADMAP.md.

    This modifies the roadmap file to:
    - Comment out the failed original task
    - Insert the rewritten/decomposed replacement tasks
    - Add audit trail comment

    For DEFER strategy, adds task to deferred_tasks list for later retry.
    For SKIP/ESCALATE strategy, adds task to failed_tasks list.

    Args:
        state: The executor graph state with recovery_proposal.
        roadmap_reader: Optional callable for reading roadmap (testing).
        roadmap_writer: Optional callable for writing roadmap (testing).

    Returns:
        Updated state with:
        - tasks: Updated task queue with replacement tasks
        - roadmap_modified: True if roadmap was changed
        - recovery_applied: True if recovery was applied
        - deferred_tasks: List of deferred tasks (DEFER strategy)
        - failed_tasks: List of failed tasks (SKIP/ESCALATE strategy)

    Example:
        >>> state = {"recovery_proposal": proposal, "recovery_approved": True}
        >>> result = await apply_recovery_node(state)
        >>> result["recovery_applied"]
        True
    """
    proposal: RecoveryProposal | None = state.get("recovery_proposal")

    # No proposal to apply
    if proposal is None:
        logger.info("apply_recovery_skipped", reason="No recovery proposal")
        return state

    # Not approved - skip applying
    if not state.get("recovery_approved"):
        logger.info("apply_recovery_skipped", reason="Recovery not approved")
        return state

    # Handle DEFER strategy - add to deferred list
    if proposal.strategy == RecoveryStrategy.DEFER:
        deferred = list(state.get("deferred_tasks", []))
        deferred.append(
            {
                "task": proposal.original_task,
                "reason": proposal.failure_reason.value,
                "retry_count": 0,
            }
        )
        logger.info(
            "task_deferred",
            task=proposal.original_task[:50],
            reason=proposal.failure_reason.value,
        )
        return {
            **state,
            "deferred_tasks": deferred,
            "recovery_applied": True,
        }

    # Handle SKIP strategy - add to failed list, don't modify roadmap
    if proposal.strategy == RecoveryStrategy.SKIP:
        failed = list(state.get("failed_tasks", []))
        failed.append(
            {
                "task": proposal.original_task,
                "reason": proposal.failure_reason.value,
                "strategy": proposal.strategy.value,
            }
        )
        logger.info(
            "task_skipped",
            task=proposal.original_task[:50],
            reason=proposal.failure_reason.value,
        )
        return {
            **state,
            "failed_tasks": failed,
            "recovery_applied": True,
        }

    # Handle ESCALATE strategy - add to failed list with escalation flag
    if proposal.strategy == RecoveryStrategy.ESCALATE:
        failed = list(state.get("failed_tasks", []))
        failed.append(
            {
                "task": proposal.original_task,
                "reason": proposal.failure_reason.value,
                "strategy": proposal.strategy.value,
                "requires_human": True,
            }
        )
        logger.info(
            "task_escalated",
            task=proposal.original_task[:50],
            reason=proposal.failure_reason.value,
        )
        return {
            **state,
            "failed_tasks": failed,
            "recovery_applied": True,
        }

    # REWRITE or DECOMPOSE: Update ROADMAP.md
    roadmap_path = state.get("roadmap_path")

    if not roadmap_path:
        logger.warning(
            "apply_recovery_failed",
            reason="No roadmap_path in state",
        )
        return {**state, "recovery_applied": False}

    # Read roadmap content
    try:
        if roadmap_reader is not None:
            roadmap_content = roadmap_reader(roadmap_path)
        else:
            from pathlib import Path

            roadmap_content = Path(roadmap_path).read_text()
    except FileNotFoundError:
        logger.error(
            "apply_recovery_failed",
            reason="Roadmap file not found",
            path=str(roadmap_path),
        )
        return {**state, "recovery_applied": False}
    except Exception as e:
        logger.error(
            "apply_recovery_failed",
            reason="Failed to read roadmap",
            error=str(e),
        )
        return {**state, "recovery_applied": False}

    # Find and replace the original task
    updated_content, found = _replace_task_in_roadmap(
        roadmap_content,
        proposal,
    )

    if not found:
        logger.warning(
            "apply_recovery_partial",
            reason="Original task not found in roadmap",
            task=proposal.original_task[:50],
        )
        # Still apply recovery to task queue even if roadmap update fails

    # Write updated roadmap
    if found:
        try:
            if roadmap_writer is not None:
                roadmap_writer(roadmap_path, updated_content)
            else:
                from pathlib import Path

                Path(roadmap_path).write_text(updated_content)
            logger.info(
                "roadmap_updated",
                strategy=proposal.strategy.value,
                replacement_count=len(proposal.proposed_tasks),
            )
        except Exception as e:
            logger.error(
                "apply_recovery_failed",
                reason="Failed to write roadmap",
                error=str(e),
            )
            return {**state, "recovery_applied": False}

    # Add replacement tasks to the task queue
    tasks = list(state.get("tasks", []))
    # Insert in reverse order so first task ends up at index 0
    for i, task_desc in enumerate(reversed(proposal.proposed_tasks)):
        # Insert at front to execute next
        tasks.insert(
            0,
            {
                "description": task_desc,
                "is_recovery": True,
                "original_task": proposal.original_task,
                "index": len(proposal.proposed_tasks) - 1 - i,
            },
        )

    return {
        **state,
        "tasks": tasks,
        "roadmap_modified": found,
        "recovery_applied": True,
    }


def _replace_task_in_roadmap(
    content: str,
    proposal: RecoveryProposal,
) -> tuple[str, bool]:
    """Replace a task in roadmap content with recovery tasks.

    Creates an audit trail with comments showing:
    - The recovery strategy used
    - The original task that failed
    - The reason for failure
    - The replacement task(s)

    Args:
        content: The current roadmap content.
        proposal: The recovery proposal with replacement tasks.

    Returns:
        Tuple of (updated_content, was_found).
    """
    import re
    from datetime import datetime

    original_task = proposal.original_task.strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build replacement text with audit trail
    replacement_lines = [
        f"<!-- RECOVERY: {proposal.strategy.value} at {timestamp} -->",
        f"<!-- Original: {original_task} -->",
        f"<!-- Reason: {proposal.failure_reason.value} -->",
    ]
    for task in proposal.proposed_tasks:
        task_text = task.strip()
        # Ensure proper checkbox format
        if not task_text.startswith("- [ ]"):
            task_text = f"- [ ] {task_text}"
        replacement_lines.append(task_text)

    replacement = "\n".join(replacement_lines)

    # Try different patterns to find the task
    patterns = [
        # Exact match with checkbox (unchecked)
        re.escape(f"- [ ] {original_task}"),
        # Exact match with checkbox (checked)
        re.escape(f"- [x] {original_task}"),
        # Match with any leading whitespace
        r"^\s*-\s*\[\s*[xX ]?\s*\]\s*" + re.escape(original_task),
    ]

    for pattern in patterns:
        regex = re.compile(pattern, re.MULTILINE)
        if regex.search(content):
            updated = regex.sub(replacement, content, count=1)
            return updated, True

    return content, False


# =============================================================================
# Phase 2.9.6: Deferred Task Retry
# =============================================================================

# Default maximum retry attempts for deferred tasks
MAX_DEFERRED_RETRIES: int = 1


async def retry_deferred_node(
    state: dict[str, Any],
    *,
    max_retries: int = MAX_DEFERRED_RETRIES,
) -> dict[str, Any]:
    """Retry deferred tasks after all other tasks complete.

    This node runs at the END of execution, not during the main loop.
    Deferred tasks are those that failed due to missing dependencies
    or environment issues that may be resolved after other tasks complete.

    Args:
        state: The executor graph state with deferred_tasks list.
        max_retries: Maximum retry attempts per task (default: 1).

    Returns:
        Updated state with:
        - tasks: Updated task queue with deferred tasks to retry
        - deferred_tasks: Updated list with incremented retry counts
        - retrying_deferred: True if any tasks are being retried

    Example:
        >>> state = {"deferred_tasks": [{"task": "Config DB", "reason": "missing_dependency"}]}
        >>> result = await retry_deferred_node(state)
        >>> result["retrying_deferred"]
        True
    """
    deferred = state.get("deferred_tasks", [])

    if not deferred:
        logger.info("retry_deferred_skipped", reason="No deferred tasks")
        return {**state, "retrying_deferred": False}

    # Find tasks eligible for retry
    tasks_to_retry = []
    updated_deferred = []

    for item in deferred:
        retry_count = item.get("retry_count", 0)

        if retry_count < max_retries:
            # Eligible for retry
            tasks_to_retry.append(
                {
                    "description": item["task"],
                    "is_deferred_retry": True,
                    "original_reason": item.get("reason", "unknown"),
                    "retry_attempt": retry_count + 1,
                }
            )
            # Update retry count
            updated_deferred.append(
                {
                    **item,
                    "retry_count": retry_count + 1,
                }
            )
        else:
            # Max retries exceeded, keep in deferred list but don't retry
            updated_deferred.append(item)
            logger.info(
                "deferred_task_max_retries",
                task=item["task"][:50],
                retry_count=retry_count,
            )

    if not tasks_to_retry:
        logger.info(
            "retry_deferred_skipped",
            reason="All deferred tasks have exceeded max retries",
        )
        return {
            **state,
            "deferred_tasks": updated_deferred,
            "retrying_deferred": False,
        }

    logger.info(
        "retrying_deferred_tasks",
        count=len(tasks_to_retry),
        total_deferred=len(deferred),
    )

    # Add to task queue
    tasks = list(state.get("tasks", []))
    tasks.extend(tasks_to_retry)

    return {
        **state,
        "tasks": tasks,
        "deferred_tasks": updated_deferred,
        "retrying_deferred": True,
    }


def should_retry_deferred(state: dict[str, Any]) -> bool:
    """Check if there are deferred tasks eligible for retry.

    Used as a routing condition in the graph to determine if
    the retry_deferred_node should be executed.

    Args:
        state: The executor graph state.

    Returns:
        True if there are deferred tasks that haven't exceeded max retries.

    Example:
        >>> state = {"tasks": [], "deferred_tasks": [{"task": "X", "retry_count": 0}]}
        >>> should_retry_deferred(state)
        True
    """
    # Only retry deferred when no more regular tasks
    if state.get("tasks"):
        return False

    deferred = state.get("deferred_tasks", [])
    if not deferred:
        return False

    # Check if any task is eligible for retry
    for item in deferred:
        if item.get("retry_count", 0) < MAX_DEFERRED_RETRIES:
            return True

    return False


# =============================================================================
# Phase 2.9.7: Audit Trail & Reporting
# =============================================================================


@dataclass
class ExecutionReport:
    """Full audit report of executor run.

    Captures comprehensive metrics about an executor run including
    task completion rates, recovery attempts, and any roadmap modifications.

    Attributes:
        started_at: When execution started.
        completed_at: When execution finished.
        total_tasks: Total number of tasks in roadmap.
        completed_tasks: Number of successfully completed tasks.
        failed_tasks: Number of tasks that failed permanently.
        deferred_tasks: Number of tasks deferred for later.
        recoveries_attempted: Total recovery attempts made.
        recoveries_successful: Number of successful recoveries.
        roadmap_modifications: List of changes made to ROADMAP.md.
        errors: List of error details for debugging.

    Example:
        >>> report = ExecutionReport(
        ...     started_at=datetime.now(),
        ...     completed_at=datetime.now(),
        ...     total_tasks=10,
        ...     completed_tasks=8,
        ...     failed_tasks=1,
        ...     deferred_tasks=1,
        ... )
        >>> print(report.success_rate)
        0.8
    """

    started_at: Any  # datetime
    completed_at: Any  # datetime
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    deferred_tasks: int = 0
    recoveries_attempted: int = 0
    recoveries_successful: int = 0
    roadmap_modifications: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate task success rate as a decimal."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    @property
    def duration_seconds(self) -> float:
        """Calculate execution duration in seconds."""
        if self.started_at is None or self.completed_at is None:
            return 0.0
        # Handle both datetime objects and strings
        try:
            return (self.completed_at - self.started_at).total_seconds()
        except TypeError:
            # Timestamps are strings or incompatible types
            return 0.0

    def to_markdown(self) -> str:
        """Generate human-readable report.

        Returns:
            Markdown-formatted execution report.
        """
        from datetime import timedelta

        duration = timedelta(seconds=self.duration_seconds)
        success_pct = self.success_rate * 100

        modifications_text = self._format_modifications()
        errors_text = self._format_errors()

        return f"""# Execution Report

**Duration**: {duration}
**Success Rate**: {success_pct:.1f}%

## Summary
| Metric | Count |
|--------|-------|
| Total Tasks | {self.total_tasks} |
| Completed | {self.completed_tasks} |
| Failed | {self.failed_tasks} |
| Deferred | {self.deferred_tasks} |

## Recoveries
| Metric | Count |
|--------|-------|
| Attempted | {self.recoveries_attempted} |
| Successful | {self.recoveries_successful} |

## Roadmap Modifications
{modifications_text}

## Errors
{errors_text}
"""

    def _format_modifications(self) -> str:
        """Format roadmap modifications for markdown output."""
        if not self.roadmap_modifications:
            return "_No modifications made._"

        lines = []
        for mod in self.roadmap_modifications:
            strategy = mod.get("strategy", "unknown")
            original = mod.get("original_task", "Unknown task")
            replacements = mod.get("replacement_count", 0)
            lines.append(f"- **{strategy}**: `{original}` → {replacements} task(s)")

        return "\n".join(lines)

    def _format_errors(self) -> str:
        """Format errors for markdown output."""
        if not self.errors:
            return "_No errors recorded._"

        lines = []
        for error in self.errors:
            task = error.get("task", "Unknown task")
            message = error.get("message", "No message")
            lines.append(f"- **{task}**: {message}")

        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Generate machine-readable report for CI/CD.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "started_at": (
                self.started_at.isoformat()
                if hasattr(self.started_at, "isoformat")
                else str(self.started_at)
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if hasattr(self.completed_at, "isoformat")
                else str(self.completed_at)
            ),
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
            "tasks": {
                "total": self.total_tasks,
                "completed": self.completed_tasks,
                "failed": self.failed_tasks,
                "deferred": self.deferred_tasks,
            },
            "recoveries": {
                "attempted": self.recoveries_attempted,
                "successful": self.recoveries_successful,
            },
            "roadmap_modifications": self.roadmap_modifications,
            "errors": self.errors,
        }


async def generate_report_node(
    state: dict[str, Any],
    *,
    report_dir: str | None = None,
    report_writer: callable | None = None,
) -> dict[str, Any]:
    """Generate execution report at end of run.

    Creates both human-readable markdown and machine-readable JSON
    reports summarizing the executor run.

    Args:
        state: The executor graph state with execution metrics.
        report_dir: Directory to write reports (default: .executor/).
        report_writer: Optional callable for writing files (testing).

    Returns:
        Updated state with:
        - execution_report: The ExecutionReport object
        - report_path: Path to the generated report files

    Example:
        >>> result = await generate_report_node(state)
        >>> result["execution_report"].success_rate
        0.8
    """
    from datetime import datetime

    # Extract metrics from state
    started_at = state.get("started_at", datetime.now())
    completed_at = datetime.now()

    # Count tasks
    total_tasks = state.get("total_task_count", 0)
    completed_list = state.get("completed_tasks", [])
    failed_list = state.get("failed_tasks", [])
    deferred_list = state.get("deferred_tasks", [])

    completed_count = len(completed_list) if isinstance(completed_list, list) else 0
    failed_count = len(failed_list) if isinstance(failed_list, list) else 0
    deferred_count = len(deferred_list) if isinstance(deferred_list, list) else 0

    # If total not tracked, infer from lists
    if total_tasks == 0:
        total_tasks = completed_count + failed_count + deferred_count

    # Recovery metrics
    recoveries_attempted = state.get("recoveries_attempted", 0)
    recoveries_successful = state.get("recoveries_successful", 0)

    # Collect roadmap modifications
    modifications = state.get("roadmap_modifications", [])
    if not modifications:
        modifications = []

    # Collect errors
    errors = []
    for failed in failed_list if isinstance(failed_list, list) else []:
        errors.append(
            {
                "task": failed.get("task", "Unknown"),
                "message": failed.get("reason", "Unknown error"),
                "strategy": failed.get("strategy", "none"),
            }
        )

    # Create report
    report = ExecutionReport(
        started_at=started_at,
        completed_at=completed_at,
        total_tasks=total_tasks,
        completed_tasks=completed_count,
        failed_tasks=failed_count,
        deferred_tasks=deferred_count,
        recoveries_attempted=recoveries_attempted,
        recoveries_successful=recoveries_successful,
        roadmap_modifications=modifications,
        errors=errors,
    )

    # Determine output directory
    if report_dir is None:
        report_dir = state.get("report_dir", ".executor")

    # Write reports
    try:
        if report_writer is not None:
            # Use provided writer (for testing)
            report_writer(report_dir, report)
        else:
            _write_reports(report_dir, report)

        logger.info(
            "execution_report_generated",
            report_dir=report_dir,
            success_rate=f"{report.success_rate:.1%}",
            total_tasks=report.total_tasks,
            completed=report.completed_tasks,
            failed=report.failed_tasks,
        )
    except Exception as e:
        logger.error(
            "report_generation_failed",
            error=str(e),
        )

    return {
        **state,
        "execution_report": report,
        "report_path": report_dir,
        "completed_at": completed_at,
    }


def _write_reports(report_dir: str, report: ExecutionReport) -> None:
    """Write report files to disk.

    Creates the report directory if it doesn't exist and writes
    both markdown and JSON format reports.

    Args:
        report_dir: Directory to write reports.
        report: The ExecutionReport to write.
    """
    import json
    from pathlib import Path

    dir_path = Path(report_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Write markdown report
    md_path = dir_path / "report.md"
    md_path.write_text(report.to_markdown())

    # Write JSON report
    json_path = dir_path / "report.json"
    json_path.write_text(json.dumps(report.to_json(), indent=2))
