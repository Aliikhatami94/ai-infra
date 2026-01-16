"""TestWriter subagent for creating comprehensive test files.

Phase 16.5.11.1 of EXECUTOR_5.md: Specialized agent for creating test suites
rather than running them.

The TestWriterAgent handles:
- Creating comprehensive test files
- Test edge cases and error paths
- Following pytest conventions
- Using proper test patterns (AAA)

This is distinct from TesterAgent which RUNS tests - TestWriterAgent CREATES them.

Example:
    ```python
    from ai_infra.executor.agents import TestWriterAgent

    agent = TestWriterAgent()
    result = await agent.execute(task, context)
    print(f"Created: {result.files_created}")
    ```
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import MetricsCallbacks
from ai_infra.executor.agents.base import (
    TEST_WRITER_SYSTEM_PROMPT,
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

__all__ = ["TestWriterAgent"]

logger = get_logger("executor.agents.testwriter")


# =============================================================================
# TestWriter Prompt Template
# =============================================================================

TEST_WRITER_PROMPT = """You are an expert test engineer responsible for writing comprehensive tests.

## Task Context
Task: {task_title}
Description: {task_description}

## Project Information
- Workspace: {workspace}
- Project Type: {project_type}
- Dependencies: {dependencies}

## Files Already Created
{files_modified}
{enriched_context}
## Instructions

You MUST create comprehensive test files using shell commands:

1. **Read the source file first** to understand the API:
   ```bash
   cat <source_file.py>
   ```

2. **Create test files** using heredoc (RECOMMENDED):
   ```bash
   cat > tests/test_module.py << 'EOF'
   \"\"\"Tests for module.\"\"\"

   import pytest
   from module import function_to_test


   class TestFunction:
       \"\"\"Tests for function_to_test.\"\"\"

       def test_basic_usage(self):
           \"\"\"Test normal usage.\"\"\"
           result = function_to_test("input")
           assert result == expected

       def test_edge_case_empty(self):
           \"\"\"Test with empty input.\"\"\"
           result = function_to_test("")
           assert result == ""

       def test_error_handling(self):
           \"\"\"Test error is raised for invalid input.\"\"\"
           with pytest.raises(ValueError):
               function_to_test(None)
   EOF
   ```

3. **Run tests to verify** they work:
   ```bash
   pytest tests/test_module.py -v
   ```

## Test Writing Principles

1. **Arrange-Act-Assert**: Setup, execute, verify
2. **One assertion per test** when possible
3. **Descriptive names**: `test_<function>_<scenario>`
4. **Docstrings**: Explain what each test verifies
5. **Edge cases**: Empty, None, type errors, boundaries
6. **Error paths**: Test exception handling with pytest.raises

## Python Testing Patterns

```python
# Basic test
def test_function_returns_expected():
    result = function("input")
    assert result == "expected"

# Exception testing
def test_function_raises_on_invalid():
    with pytest.raises(ValueError, match="specific message"):
        function(invalid_input)

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    ("a", "A"),
    ("b", "B"),
])
def test_function_multiple_inputs(input, expected):
    assert function(input) == expected

# Fixtures
@pytest.fixture
def sample_data():
    return {{"key": "value"}}

def test_with_fixture(sample_data):
    result = process(sample_data)
    assert result is not None
```

## File Writing Best Practices (CRITICAL)

**ALWAYS use heredoc for multi-line files:**
```bash
cat > filename.py << 'EOF'
<actual code with real newlines>
EOF
```

**NEVER use these patterns (they create broken files):**
- `echo "line1\\nline2"` - writes literal \\n, not actual newlines
- `echo -e "line1\\nline2"` - not portable, may fail in bash

## Quality Standards

- Test ALL public functions and methods
- Include both happy path and error cases
- Use proper type hints in test code
- Follow existing test patterns in the project
- Ensure tests pass before completing
"""


# =============================================================================
# TestWriterAgent
# =============================================================================


@SubAgentRegistry.register(SubAgentType.TESTWRITER)
class TestWriterAgent(SubAgent):
    """Specialized agent for creating comprehensive test files.

    Unlike TesterAgent which runs tests, TestWriterAgent creates them.
    Optimized for test file creation with proper coverage.
    """

    name = "TestWriter"
    description = "Creates comprehensive test suites"
    model = "claude-sonnet-4-20250514"
    system_prompt = TEST_WRITER_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 300.0,
        shell_timeout: float = 60.0,
    ) -> None:
        """Initialize the test writer agent.

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
        """Execute a test writing task.

        Args:
            task: The task to execute.
            context: Execution context.

        Returns:
            SubAgentResult with execution outcome.
        """
        start_time = time.perf_counter()
        files_created: list[str] = []
        files_modified: list[str] = []

        logger.info(f"TestWriterAgent executing task: {task.title}")

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

            # Format the prompt with enriched context
            deps = ", ".join(context.dependencies) if context.dependencies else "None"
            files = (
                "\n".join(f"- {f}" for f in context.files_modified)
                if context.files_modified
                else "None"
            )

            # Include enriched context in prompt
            enriched_context = context.format_for_prompt()

            # Phase 16.5.11.5.4: Add source file previews for test writing context
            source_previews = context.format_source_previews()
            if source_previews:
                enriched_context = enriched_context + "\n\n" + source_previews

            if enriched_context:
                enriched_context = "\n" + enriched_context + "\n"

            prompt = TEST_WRITER_PROMPT.format(
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
                f"Create comprehensive tests for: {task.title}\n\n{task.description or ''}"
            )

            # Extract output text
            output = self._extract_output(result)

            # Analyze command history for file changes
            files_created, files_modified = self._analyze_file_changes(session.command_history)

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Get token metrics from callback
            llm_metrics = metrics_cb.get_summary().get("llm", {})

            logger.info(
                f"TestWriterAgent completed in {duration_ms:.0f}ms: "
                f"{len(files_created)} created, {len(files_modified)} modified, "
                f"{llm_metrics.get('total_tokens', 0)} tokens"
            )

            return SubAgentResult(
                success=True,
                output=output,
                files_created=files_created,
                files_modified=files_modified,
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
            logger.warning(f"TestWriterAgent timed out: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=f"Execution timed out after {self._timeout}s",
                metrics={"duration_ms": duration_ms},
            )

        except Exception as e:
            logger.exception(f"TestWriterAgent failed: {e}")
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

        Args:
            command_history: List of executed commands (strings or ShellResult).

        Returns:
            Tuple of (files_created, files_modified).
        """
        files_created: set[str] = set()
        files_modified: set[str] = set()

        for cmd in command_history:
            cmd_str = self._extract_command_string(cmd)

            # Detect file creation with cat
            if "cat >" in cmd_str or "cat>" in cmd_str:
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
            elif cmd_str.startswith("mkdir "):
                continue  # Directories, not files

            # Detect sed edits
            elif "sed " in cmd_str and " -i" in cmd_str:
                parts = cmd_str.split()
                if parts:
                    filename = parts[-1]
                    if not filename.startswith("-"):
                        files_modified.add(filename)

        return list(files_created), list(files_modified)

    def _extract_command_string(self, cmd: str | ShellResult) -> str:
        """Extract command string from command history entry.

        Args:
            cmd: Command entry (string or ShellResult).

        Returns:
            Command string.
        """
        if isinstance(cmd, str):
            return cmd
        elif hasattr(cmd, "command"):
            return str(cmd.command)
        return str(cmd)
