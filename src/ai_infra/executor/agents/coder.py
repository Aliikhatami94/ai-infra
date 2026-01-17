"""Coder subagent for writing and editing code.

Phase 3.3.3 of EXECUTOR_1.md: Specialized agent for code creation
and modification tasks.

Phase 16.5.10: Added file writing best practices to prevent literal \\n issues.

The CoderAgent handles tasks like:
- Creating new files
- Editing existing code
- Implementing features
- Fixing bugs
- Refactoring code

Example:
    ```python
    from ai_infra.executor.agents import CoderAgent

    agent = CoderAgent()
    result = await agent.execute(task, context)
    ```
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import MetricsCallbacks
from ai_infra.executor.agents.base import (
    CODER_SYSTEM_PROMPT,
    ExecutionContext,
    SubAgent,
    SubAgentResult,
)
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.llm.agent import Agent
from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.tool import create_shell_tool, set_current_session
from ai_infra.llm.shell.types import ShellResult
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

__all__ = ["CoderAgent"]

logger = get_logger("executor.agents.coder")


# =============================================================================
# Coder Prompt Template
# =============================================================================

CODER_PROMPT = """You are an expert software engineer implementing code changes.

## Task
Title: {task_title}
Description: {task_description}

## Project Context
- Workspace: {workspace}
- Project Type: {project_type}
- Dependencies: {dependencies}

## Previously Modified Files
{files_modified}
{enriched_context}
## Instructions

You MUST implement the requested changes using shell commands:

1. **Read existing code** before making changes:
   ```bash
   cat <filename>
   ```

2. **Create new files** using heredoc (RECOMMENDED):
   ```bash
   cat > filename.py << 'EOF'
   import argparse

   def main():
       parser = argparse.ArgumentParser()
       # Implementation here
       pass

   if __name__ == "__main__":
       main()
   EOF
   ```

3. **Edit existing files** using sed or full replacement:
   ```bash
   # For small changes, use sed
   sed -i '' 's/old_pattern/new_pattern/g' filename

   # For larger changes, rewrite the entire file with heredoc
   cat > filename << 'EOF'
   <complete new content>
   EOF
   ```

4. **Verify changes** after making them:
   ```bash
   cat <filename>  # Verify content
   python -m py_compile <file.py>  # Check syntax
   ```

## File Writing Best Practices (CRITICAL)

**ALWAYS use heredoc for multi-line files:**
```bash
cat > filename.py << 'EOF'
<actual code with real newlines>
EOF
```

**For simple single-line content:**
```bash
printf '%s\\n' 'content' > filename.txt
```

**NEVER use these patterns (they create broken files):**
- `echo "line1\\nline2"` - writes literal \\n, not actual newlines
- `echo -e "line1\\nline2"` - not portable, may fail in bash
- Single-line echo with escaped newlines

## Quality Standards

- Follow existing code conventions
- Add proper type hints and docstrings (Python)
- Include error handling
- Match the style of surrounding code
- Ensure code compiles/parses correctly

## Important

- Work incrementally - make one change at a time
- Always read the file BEFORE editing to understand context
- Verify each change works before moving on
- Report any files you create or modify
"""


# =============================================================================
# CoderAgent
# =============================================================================


@SubAgentRegistry.register(SubAgentType.CODER)
class CoderAgent(SubAgent):
    """Specialized agent for writing and editing code.

    Uses shell commands to read, create, and edit files.
    Optimized for code creation and modification tasks.
    """

    name = "Coder"
    description = "Writes and edits code using shell commands"
    model = "claude-sonnet-4-20250514"
    system_prompt = CODER_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 300.0,
        shell_timeout: float = 60.0,
    ) -> None:
        """Initialize the coder agent.

        Args:
            model: Optional model override.
            timeout: Maximum execution time.
            shell_timeout: Timeout for shell commands.
        """
        super().__init__(model=model)
        self._timeout = timeout
        self._shell_timeout = shell_timeout

    def _get_tools(self) -> list[Any]:
        """Get tools for this agent.

        Note: Tools are created dynamically per-execution with
        the appropriate shell session. This returns an empty list.
        """
        return []

    async def execute(
        self,
        task: TodoItem,
        context: ExecutionContext,
    ) -> SubAgentResult:
        """Execute a coding task.

        Args:
            task: The task to execute.
            context: Execution context.

        Returns:
            SubAgentResult with execution outcome.
        """
        start_time = time.perf_counter()
        files_created: list[str] = []
        files_modified: list[str] = []

        logger.info(f"CoderAgent executing task: {task.title}")

        # Create shell session for this execution
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
            deps = ", ".join(context.dependencies) if context.dependencies else "None"
            files = (
                "\n".join(f"- {f}" for f in context.files_modified)
                if context.files_modified
                else "None"
            )

            # Phase 16.5.5.10: Include enriched context in prompt
            enriched_context = context.format_for_prompt()

            # Phase 16.5.11.5.3: Add code patterns to help CoderAgent follow conventions
            patterns_summary = context.get_patterns_summary()
            if patterns_summary:
                enriched_context = enriched_context + "\n\n" + patterns_summary

            if enriched_context:
                enriched_context = "\n" + enriched_context + "\n"

            prompt = CODER_PROMPT.format(
                task_title=task.title,
                task_description=task.description or "No description provided.",
                workspace=str(context.workspace),
                project_type=context.project_type,
                dependencies=deps,
                files_modified=files,
                enriched_context=enriched_context,
            )

            # Create metrics callback to track token usage
            metrics_cb = MetricsCallbacks()

            # Create agent with shell tool and callbacks
            agent = Agent(
                tools=[shell_tool],
                model_name=self._model,
                system=prompt,
                callbacks=metrics_cb,
            )

            # Run the task
            result = await agent.arun(
                f"Implement the following: {task.title}\n\n{task.description or ''}"
            )

            # Extract output text
            output = self._extract_output(result)

            # Analyze command history for file changes
            files_created, files_modified = self._analyze_file_changes(session.command_history)

            # Phase 16.5.10: Validate created files and attempt repair if needed
            all_files = files_created + files_modified
            validation_errors = self._validate_created_files(context.workspace, all_files)

            # Attempt to repair files with malformed newlines
            repaired_files: list[str] = []
            for file in all_files:
                filepath = context.workspace / file
                if filepath.exists() and self._repair_newlines(filepath):
                    repaired_files.append(file)

            # Re-validate after repair
            if repaired_files:
                validation_errors = self._validate_created_files(context.workspace, all_files)
                logger.info(f"Repaired {len(repaired_files)} files with malformed newlines")

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Get token metrics from callback
            llm_metrics = metrics_cb.get_summary().get("llm", {})

            logger.info(
                f"CoderAgent completed in {duration_ms:.0f}ms: "
                f"{len(files_created)} created, {len(files_modified)} modified, "
                f"{llm_metrics.get('total_tokens', 0)} tokens"
            )

            # Log any remaining validation errors
            if validation_errors:
                logger.warning(
                    "File validation issues after repair: %s",
                    "; ".join(validation_errors),
                )

            return SubAgentResult(
                success=True,
                output=output,
                files_created=files_created,
                files_modified=files_modified,
                metrics={
                    "duration_ms": duration_ms,
                    "commands_run": len(session.command_history),
                    "tokens_in": llm_metrics.get("total_tokens", 0),  # Approx, not split
                    "tokens_out": 0,  # Not available separately
                    "total_tokens": llm_metrics.get("total_tokens", 0),
                    "model": self._model,
                    "agent_type": self.name,
                    "llm_calls": llm_metrics.get("calls", 0),
                    "files_repaired": len(repaired_files),
                    "validation_errors": len(validation_errors),
                },
            )

        except TimeoutError as e:
            logger.warning(f"CoderAgent timed out: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=f"Execution timed out after {self._timeout}s",
                metrics={"duration_ms": duration_ms},
            )

        except Exception as e:
            logger.exception(f"CoderAgent failed: {e}")
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
        """Extract text output from agent result.

        Args:
            result: Agent response.

        Returns:
            String output.
        """
        if hasattr(result, "content"):
            return str(result.content)
        elif isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return str(result.get("content", result.get("output", str(result))))
        return str(result)

    def _analyze_file_changes(
        self,
        command_history: list[str | ShellResult],
    ) -> tuple[list[str], list[str]]:
        """Analyze command history to detect file changes.

        Phase 16.5.2: Updated to handle both string commands and ShellResult objects.

        Args:
            command_history: List of executed commands (strings or ShellResult).

        Returns:
            Tuple of (files_created, files_modified).
        """
        files_created: set[str] = set()
        files_modified: set[str] = set()

        for cmd in command_history:
            # Phase 16.5.2: Extract command string from ShellResult or use string directly
            cmd_str = self._extract_command_string(cmd)
            cmd_lower = cmd_str.lower().strip()

            # Detect file creation with cat
            if "cat >" in cmd_str or "cat>" in cmd_str:
                # Extract filename after 'cat >'
                parts = cmd_str.split(">", 1)
                if len(parts) > 1:
                    filename = parts[1].split("<<")[0].strip().split()[0]
                    files_created.add(filename)

            # Detect touch
            elif cmd_str.startswith("touch "):
                filename = cmd_str.split()[1] if len(cmd_str.split()) > 1 else ""
                if filename:
                    files_created.add(filename)

            # Detect mkdir
            elif "mkdir" in cmd_lower:
                # Could track directories too
                pass

            # Detect sed -i (in-place edit)
            elif "sed -i" in cmd_lower:
                # Extract filename (last argument typically)
                parts = cmd_str.split()
                if parts:
                    filename = parts[-1]
                    if not filename.startswith("-") and ("/" in filename or "." in filename):
                        files_modified.add(filename)

            # Detect echo redirect
            elif "echo" in cmd_lower and ">" in cmd_str:
                parts = cmd_str.split(">")
                if len(parts) > 1:
                    filename = parts[-1].strip().split()[0]
                    if ">>" in cmd_str:
                        files_modified.add(filename)
                    else:
                        files_created.add(filename)

        return list(files_created), list(files_modified)

    def _extract_command_string(self, cmd: str | ShellResult) -> str:
        """Extract command string from various input types.

        Phase 16.5.2: Defensive helper to handle mixed command history types.

        Args:
            cmd: Either a string command or ShellResult object.

        Returns:
            The command as a string.
        """
        if isinstance(cmd, ShellResult):
            return cmd.command
        if isinstance(cmd, str):
            return cmd
        # Fallback for dict-like objects or other types
        if hasattr(cmd, "command"):
            return str(cmd.command)
        if hasattr(cmd, "get"):
            return str(cmd.get("command", str(cmd)))
        return str(cmd)

    def _validate_created_files(
        self,
        workspace: Path,
        files: list[str],
    ) -> list[str]:
        """Validate syntax of created files and detect malformed newlines.

        Phase 16.5.10: Detect files with literal \\n instead of real newlines,
        which can happen when LLMs use `echo` incorrectly.

        Args:
            workspace: Workspace root path.
            files: List of file paths relative to workspace.

        Returns:
            List of error messages for files with issues.
        """
        errors: list[str] = []

        for file in files:
            filepath = workspace / file
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text()

                # Check for literal \n that should be newlines
                # A valid Python file should have real newlines, not literal \n
                if self._has_malformed_newlines(content):
                    errors.append(
                        f"{file}: Contains literal \\n instead of newlines "
                        "(file may have been created incorrectly)"
                    )
                    logger.warning(
                        f"File {file} may have malformed newlines (literal \\n detected)"
                    )

                # For Python files, try to compile to check syntax
                if filepath.suffix == ".py":
                    try:
                        compile(content, str(filepath), "exec")
                    except SyntaxError as e:
                        errors.append(f"{file}: Syntax error - {e.msg} (line {e.lineno})")

            except (OSError, UnicodeDecodeError) as e:
                errors.append(f"{file}: Could not read file - {e}")

        return errors

    def _has_malformed_newlines(self, content: str) -> bool:
        """Check if content has literal \\n instead of real newlines.

        Phase 16.5.10: Detect malformed files created by incorrect shell commands.

        The heuristic: if we see literal "\\n" in the content but the content
        has very few actual newlines (less than expected), it's likely malformed.

        Args:
            content: File content to check.

        Returns:
            True if content appears to have malformed newlines.
        """
        # Check for literal \n (escaped backslash-n)
        literal_newline_count = content.count("\\n")
        real_newline_count = content.count("\n")

        # If there are literal \n and very few real newlines, it's malformed
        # A properly formatted multi-line file should have more real newlines
        if literal_newline_count > 0:
            # If there are more literal \n than real newlines, definitely malformed
            if literal_newline_count > real_newline_count:
                return True
            # If the content is long but has few newlines and many literal \n
            if len(content) > 100 and real_newline_count < 3 and literal_newline_count > 2:
                return True

        return False

    def _repair_newlines(self, filepath: Path) -> bool:
        """Attempt to repair literal \\n in file content.

        Phase 16.5.10: Fix files that were created with incorrect echo commands.

        Args:
            filepath: Path to the file to repair.

        Returns:
            True if file was modified, False otherwise.
        """
        try:
            content = filepath.read_text()

            if not self._has_malformed_newlines(content):
                return False

            # Handle literal \r\n (Windows newlines written incorrectly) FIRST
            # before replacing \n, otherwise \r\n becomes \r followed by real newline
            repaired = content.replace("\\r\\n", "\n")
            # Handle standalone literal \r (carriage return)
            repaired = repaired.replace("\\r", "\r")
            # Replace literal \n with actual newlines
            repaired = repaired.replace("\\n", "\n")
            # Also handle literal \t (tabs)
            repaired = repaired.replace("\\t", "\t")

            # Only write if content changed
            if repaired != content:
                filepath.write_text(repaired)
                logger.info(f"Repaired malformed newlines in {filepath}")
                return True

        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Could not repair file {filepath}: {e}")

        return False
