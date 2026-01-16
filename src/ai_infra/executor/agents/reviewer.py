"""Reviewer subagent for code review.

Phase 3.3.4 of EXECUTOR_1.md: Specialized agent for reviewing
code changes and providing feedback.

The ReviewerAgent handles:
- Code quality review
- Security vulnerability detection
- Performance issue identification
- Style and convention checks
- Test coverage suggestions

Example:
    ```python
    from ai_infra.executor.agents import ReviewerAgent

    agent = ReviewerAgent()
    result = await agent.execute(task, context)
    print(result.verdict)  # "APPROVE" or "REQUEST_CHANGES"
    ```
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import MetricsCallbacks
from ai_infra.executor.agents.base import (
    REVIEWER_SYSTEM_PROMPT,
    ExecutionContext,
    SubAgent,
    SubAgentResult,
)
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.llm.agent import Agent
from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.tool import create_shell_tool, set_current_session
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

__all__ = ["ReviewerAgent"]

logger = get_logger("executor.agents.reviewer")


# =============================================================================
# Reviewer Prompt Template
# =============================================================================

REVIEWER_PROMPT = """You are a senior code reviewer examining recent changes.

## Task Context
Task: {task_title}
Description: {task_description}

## Files to Review
{files_to_review}

## Project Context
- Workspace: {workspace}
- Project Type: {project_type}
{enriched_context}
## Your Review Process

1. **Read each modified file** to understand the changes:
   ```bash
   cat <filename>
   ```

2. **Check for issues** in these categories:
   - **Bugs**: Logic errors, null checks, edge cases
   - **Security**: SQL injection, XSS, hardcoded secrets
   - **Performance**: N+1 queries, unnecessary loops
   - **Style**: Naming, formatting, documentation

3. **Run static analysis** if available:
   ```bash
   # Python
   ruff check <file>
   mypy <file>

   # JavaScript/TypeScript
   npx eslint <file>
   npx tsc --noEmit
   ```

4. **Verify tests exist** for the changes:
   ```bash
   find . -name "test_*.py" | xargs grep -l "<function_name>"
   ```

## Output Format

After reviewing, output your assessment in this format:

### Summary
Brief description of what was changed.

### Issues Found

#### Critical
- [FILE:LINE] Description of critical issue

#### Major
- [FILE:LINE] Description of major issue

#### Minor
- [FILE:LINE] Description of minor issue

### Suggestions
- Improvement suggestions not blocking approval

### Verdict
APPROVE or REQUEST_CHANGES

(Use APPROVE if no critical/major issues, REQUEST_CHANGES otherwise)
"""


# =============================================================================
# ReviewerAgent
# =============================================================================


@SubAgentRegistry.register(SubAgentType.REVIEWER)
class ReviewerAgent(SubAgent):
    """Specialized agent for code review.

    Reviews code changes for bugs, security issues,
    performance problems, and style violations.
    """

    name = "Reviewer"
    description = "Reviews code changes and provides feedback"
    model = "claude-sonnet-4-20250514"
    system_prompt = REVIEWER_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 300.0,
        shell_timeout: float = 60.0,
    ) -> None:
        """Initialize the reviewer agent.

        Args:
            model: Optional model override.
            timeout: Maximum execution time.
            shell_timeout: Timeout for shell commands.
        """
        super().__init__(model=model)
        self._timeout = timeout
        self._shell_timeout = shell_timeout

    def _get_tools(self) -> list[Any]:
        """Get tools for this agent."""
        return []

    async def execute(
        self,
        task: TodoItem,
        context: ExecutionContext,
    ) -> SubAgentResult:
        """Execute a code review task.

        Args:
            task: The review task.
            context: Execution context with files to review.

        Returns:
            SubAgentResult with review comments and verdict.
        """
        start_time = time.perf_counter()

        logger.info(f"ReviewerAgent reviewing: {task.title}")

        # Determine files to review
        files_to_review = context.files_modified or []

        if not files_to_review:
            logger.warning("No files to review")
            return SubAgentResult(
                success=True,
                output="No files to review.",
                verdict="APPROVE",
                review_comments=["No code changes to review."],
            )

        # Create shell session
        session_config = SessionConfig(workspace_root=context.workspace)
        session = ShellSession(session_config)

        try:
            await session.start()
            set_current_session(session)

            # Create shell tool
            shell_tool = create_shell_tool(
                session=session,
                default_timeout=self._shell_timeout,
            )

            # Format the prompt with enriched context (Phase 16.5.5)
            # Phase 16.5.5.10: Include enriched context in prompt
            enriched_context = context.format_for_prompt()
            if enriched_context:
                enriched_context = "\n" + enriched_context + "\n"

            prompt = REVIEWER_PROMPT.format(
                task_title=task.title,
                task_description=task.description or "No description provided.",
                workspace=str(context.workspace),
                project_type=context.project_type,
                files_to_review="\n".join(f"- {f}" for f in files_to_review),
                enriched_context=enriched_context,
            )

            # Create metrics callback to track token usage
            metrics_cb = MetricsCallbacks()

            # Create agent with callbacks
            agent = Agent(
                tools=[shell_tool],
                model_name=self._model,
                system=prompt,
                callbacks=metrics_cb,
            )

            # Run the review
            result = await agent.arun(
                f"Review the following files for the task: {task.title}\n\n"
                f"Files: {', '.join(files_to_review)}"
            )

            # Extract and parse output
            output = self._extract_output(result)
            verdict, comments = self._parse_review(output)

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Get token metrics from callback
            llm_metrics = metrics_cb.get_summary().get("llm", {})

            logger.info(
                f"ReviewerAgent completed in {duration_ms:.0f}ms: verdict={verdict}, "
                f"{llm_metrics.get('total_tokens', 0)} tokens"
            )

            return SubAgentResult(
                success=True,
                output=output,
                verdict=verdict,
                review_comments=comments,
                metrics={
                    "duration_ms": duration_ms,
                    "files_reviewed": len(files_to_review),
                    "tokens_in": llm_metrics.get("total_tokens", 0),
                    "tokens_out": 0,
                    "total_tokens": llm_metrics.get("total_tokens", 0),
                    "model": self._model,
                    "agent_type": self.name,
                    "llm_calls": llm_metrics.get("calls", 0),
                },
            )

        except TimeoutError as e:
            logger.warning(f"ReviewerAgent timed out: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=f"Review timed out after {self._timeout}s",
                verdict="REQUEST_CHANGES",
                review_comments=["Review could not be completed - timed out."],
                metrics={"duration_ms": duration_ms},
            )

        except Exception as e:
            logger.exception(f"ReviewerAgent failed: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=str(e),
                verdict="REQUEST_CHANGES",
                review_comments=[f"Review failed: {e}"],
                metrics={"duration_ms": duration_ms},
            )

        finally:
            set_current_session(None)
            await session.close()

    def _extract_output(self, result: Any) -> str:
        """Extract text output from agent result."""
        if hasattr(result, "content"):
            return str(result.content)
        elif isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return str(result.get("content", result.get("output", str(result))))
        return str(result)

    def _parse_review(self, output: str) -> tuple[str, list[str]]:
        """Parse review output to extract verdict and comments.

        Args:
            output: Raw review output.

        Returns:
            Tuple of (verdict, list of comments).
        """
        output_lower = output.lower()

        # Extract verdict
        if "request_changes" in output_lower or "request changes" in output_lower:
            verdict = "REQUEST_CHANGES"
        elif "approve" in output_lower:
            verdict = "APPROVE"
        else:
            # Default to approve if no critical issues mentioned
            if "critical" in output_lower and "]" in output_lower:
                verdict = "REQUEST_CHANGES"
            else:
                verdict = "APPROVE"

        # Extract comments (lines that look like issues)
        comments: list[str] = []
        lines = output.split("\n")

        in_issues_section = False
        for line in lines:
            stripped = line.strip()

            # Track if we're in an issues section
            if "issues found" in stripped.lower():
                in_issues_section = True
                continue
            elif "suggestions" in stripped.lower() or "verdict" in stripped.lower():
                in_issues_section = False
                continue

            # Capture issue lines
            if in_issues_section and stripped.startswith("-"):
                comments.append(stripped.lstrip("- "))
            elif stripped.startswith("[") and "]" in stripped:
                # Lines like [FILE:LINE] description
                comments.append(stripped)

        return verdict, comments
