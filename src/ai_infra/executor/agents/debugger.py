"""Debugger subagent for fixing failures.

Phase 3.3 of EXECUTOR_1.md: Specialized agent for analyzing
and fixing test failures, bugs, and errors.

The DebuggerAgent handles:
- Analyzing error messages
- Finding root causes
- Implementing fixes
- Verifying fixes work

Example:
    ```python
    from ai_infra.executor.agents import DebuggerAgent

    agent = DebuggerAgent()
    result = await agent.execute(task, context)
    ```
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import MetricsCallbacks
from ai_infra.executor.agents.base import (
    DEBUGGER_SYSTEM_PROMPT,
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

__all__ = ["DebuggerAgent"]

logger = get_logger("executor.agents.debugger")


# =============================================================================
# Debugger Prompt Template
# =============================================================================

DEBUGGER_PROMPT = """You are an expert debugger analyzing failures.

## Task Context
Task: {task_title}
Description: {task_description}

## Error Information
{error_info}

## Project Context
- Workspace: {workspace}
- Project Type: {project_type}
- Files Modified: {files_modified}
{enriched_context}
## Debugging Process

1. **Understand the error**:
   ```bash
   # Read error logs
   cat <log_file>

   # Check recent changes
   git diff HEAD~1
   ```

2. **Find the root cause**:
   ```bash
   # Read relevant source files
   cat <source_file>

   # Search for related code
   grep -r "pattern" src/
   ```

3. **Implement the fix**:
   ```bash
   # Edit the file
   cat > <filename> << 'EOF'
   <fixed content>
   EOF
   ```

4. **Verify the fix**:
   ```bash
   # Run the failing test
   pytest <test_file> -v

   # Or run all tests
   pytest -q
   ```

## Output Format

After debugging, summarize:
1. Root cause identified
2. Fix applied
3. Verification result
"""


# =============================================================================
# DebuggerAgent
# =============================================================================


@SubAgentRegistry.register(SubAgentType.DEBUGGER)
class DebuggerAgent(SubAgent):
    """Specialized agent for debugging and fixing failures.

    Analyzes error messages, identifies root causes, implements
    fixes, and verifies they work.
    """

    name = "Debugger"
    description = "Analyzes and fixes failures"
    model = "claude-sonnet-4-20250514"
    system_prompt = DEBUGGER_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 600.0,
        shell_timeout: float = 120.0,
    ) -> None:
        """Initialize the debugger agent.

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
        """Execute a debugging task.

        Args:
            task: The debug task.
            context: Execution context.

        Returns:
            SubAgentResult with debugging outcome.
        """
        start_time = time.perf_counter()

        logger.info(f"DebuggerAgent debugging: {task.title}")

        # Extract error info from context
        error_info = context.run_memory.get("error", "No error information provided")

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
            files = ", ".join(context.files_modified) if context.files_modified else "Unknown"

            # Phase 16.5.5.10: Include enriched context in prompt
            enriched_context = context.format_for_prompt()
            if enriched_context:
                enriched_context = "\n" + enriched_context + "\n"

            prompt = DEBUGGER_PROMPT.format(
                task_title=task.title,
                task_description=task.description or "Fix the failure",
                workspace=str(context.workspace),
                project_type=context.project_type,
                files_modified=files,
                error_info=error_info,
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

            # Run debugging
            result = await agent.arun(f"Debug and fix: {task.title}\n\n{task.description or ''}")

            # Extract output
            output = self._extract_output(result)

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Get token metrics from callback
            llm_metrics = metrics_cb.get_summary().get("llm", {})

            logger.info(
                f"DebuggerAgent completed in {duration_ms:.0f}ms, "
                f"{llm_metrics.get('total_tokens', 0)} tokens"
            )

            return SubAgentResult(
                success=True,
                output=output,
                metrics={
                    "duration_ms": duration_ms,
                    "commands_run": len(session.command_history),
                    "tokens_in": llm_metrics.get("total_tokens", 0),
                    "tokens_out": 0,
                    "total_tokens": llm_metrics.get("total_tokens", 0),
                    "model": self._model,
                    "agent_type": self.name,
                    "llm_calls": llm_metrics.get("calls", 0),
                },
            )

        except TimeoutError as e:
            logger.warning(f"DebuggerAgent timed out: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=f"Debugging timed out after {self._timeout}s",
                metrics={"duration_ms": duration_ms},
            )

        except Exception as e:
            logger.exception(f"DebuggerAgent failed: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=str(e),
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
