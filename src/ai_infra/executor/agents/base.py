"""Base class for specialized subagents.

Phase 3.3.2 of EXECUTOR_1.md: Provides the abstract base class that all
specialized subagents must inherit from.

Subagents are specialized agents optimized for specific tasks:
- CoderAgent: Writing and editing code
- ReviewerAgent: Reviewing code for issues
- TesterAgent: Running tests
- DebuggerAgent: Fixing failures
- ResearcherAgent: Finding information

Example:
    ```python
    from ai_infra.executor.agents.base import SubAgent, SubAgentResult
    from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType

    @SubAgentRegistry.register(SubAgentType.CODER)
    class CoderAgent(SubAgent):
        name = "Coder"
        description = "Writes and edits code"
        model = "claude-sonnet-4-20250514"

        async def execute(self, task, context):
            result = await self.agent.arun(...)
            return SubAgentResult(success=True, output=result)
    ```
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem
    from ai_infra.llm.agent import Agent

__all__ = [
    "ExecutionContext",
    "SubAgent",
    "SubAgentResult",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecutionContext:
    """Context for subagent task execution.

    Phase 16.5.5: Enhanced with rich context for better subagent code quality.
    Phase 16.5.11.5: Added completed_tasks, existing_files for improved context.

    Provides all the information a subagent needs to execute a task,
    including previous task summaries, established code patterns, and
    file contents cache.

    Attributes:
        workspace: Path to the workspace root.
        files_modified: Files already modified in this session.
        relevant_files: Files relevant to the current task.
        project_type: Detected project type (python, node, rust, etc.).
        summary: Brief summary of project context (session brief).
        run_memory: Memory from previous task executions.
        dependencies: Known project dependencies.
        previous_task_summaries: Summaries of previously completed tasks.
        established_patterns: Detected code patterns (docstring_style, etc.).
        file_contents_cache: Cache of recently created file contents.
        completed_tasks: List of completed task titles for context (Phase 16.5.11.5).
        existing_files: Full list of files existing in workspace (Phase 16.5.11.5).
        project_patterns: Detected project-wide patterns (Phase 16.5.11.5).
        source_file_previews: Content previews of source files for test writing.
    """

    workspace: Path
    files_modified: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    project_type: str = "unknown"
    summary: str = ""
    run_memory: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    # Phase 16.5.5: New fields for enriched context
    previous_task_summaries: list[str] = field(default_factory=list)
    established_patterns: dict[str, str] = field(default_factory=dict)
    file_contents_cache: dict[str, str] = field(default_factory=dict)
    # Phase 16.5.11.5: Additional context for quality improvements
    completed_tasks: list[str] = field(default_factory=list)
    existing_files: list[str] = field(default_factory=list)
    project_patterns: dict[str, Any] = field(default_factory=dict)
    source_file_previews: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace": str(self.workspace),
            "files_modified": self.files_modified,
            "relevant_files": self.relevant_files,
            "project_type": self.project_type,
            "summary": self.summary,
            "run_memory": self.run_memory,
            "dependencies": self.dependencies,
            "previous_task_summaries": self.previous_task_summaries,
            "established_patterns": self.established_patterns,
            "file_contents_cache": list(self.file_contents_cache.keys()),
            # Phase 16.5.11.5: New fields
            "completed_tasks": self.completed_tasks,
            "existing_files": self.existing_files,
            "project_patterns": self.project_patterns,
            "source_file_previews": list(self.source_file_previews.keys()),
        }

    def format_for_prompt(self) -> str:
        """Format execution context as prompt sections.

        Phase 16.5.5: Provides rich context for subagent prompts.
        Phase 16.5.11.5: Enhanced with existing files and completed tasks.

        Returns:
            Formatted string with context sections.
        """
        sections: list[str] = []

        # Code patterns section
        if self.established_patterns:
            patterns_text = "\n".join(f"- {k}: {v}" for k, v in self.established_patterns.items())
            sections.append(f"## Code Patterns to Follow\n{patterns_text}")

        # Phase 16.5.11.5: Existing files section (limit to most relevant)
        if self.existing_files:
            # Show max 15 files, prioritize source files
            source_files = [f for f in self.existing_files if f.endswith((".py", ".ts", ".js"))]
            other_files = [f for f in self.existing_files if f not in source_files]
            display_files = (source_files[:10] + other_files[:5])[:15]
            files_text = "\n".join(f"- {f}" for f in display_files)
            if len(self.existing_files) > 15:
                files_text += f"\n- ... and {len(self.existing_files) - 15} more files"
            sections.append(f"## Existing Project Files\n{files_text}")

        # Files created section
        if self.files_modified:
            files_text = "\n".join(f"- {f}" for f in self.files_modified)
            sections.append(f"## Files Created This Session\n{files_text}")

        # Phase 16.5.11.5: Completed tasks section
        if self.completed_tasks:
            tasks = "\n".join(f"- {t}" for t in self.completed_tasks[-5:])
            sections.append(f"## Previously Completed Tasks\n{tasks}")

        # Previous tasks section (limit to last 5)
        if self.previous_task_summaries:
            summaries = "\n".join(f"- {s}" for s in self.previous_task_summaries[-5:])
            sections.append(f"## Task Summaries\n{summaries}")

        # Session context
        if self.summary:
            sections.append(f"## Session Context\n{self.summary}")

        return "\n\n".join(sections)

    def format_source_previews(self, max_lines: int = 100) -> str:
        """Format source file previews for test writing context.

        Phase 16.5.11.5: Provides source code context to TestWriterAgent.

        Args:
            max_lines: Maximum lines per file preview.

        Returns:
            Formatted string with source code previews.
        """
        if not self.source_file_previews:
            return ""

        sections: list[str] = []
        for filepath, content in self.source_file_previews.items():
            lines = content.split("\n")
            if len(lines) > max_lines:
                preview = "\n".join(lines[:max_lines])
                preview += f"\n# ... ({len(lines) - max_lines} more lines)"
            else:
                preview = content

            sections.append(f"### {filepath}\n```python\n{preview}\n```")

        return "## Source Files to Test\n\n" + "\n\n".join(sections)

    def get_patterns_summary(self) -> str:
        """Get a summary of detected code patterns.

        Phase 16.5.11.5: Provides pattern context to CoderAgent.

        Returns:
            Formatted string describing detected patterns.
        """
        if not self.established_patterns and not self.project_patterns:
            return ""

        sections: list[str] = []

        if self.established_patterns:
            patterns = "\n".join(
                f"- **{k.replace('_', ' ').title()}**: {v}"
                for k, v in self.established_patterns.items()
            )
            sections.append(f"### Detected Code Patterns\n{patterns}")

        if self.project_patterns:
            proj_patterns = "\n".join(
                f"- **{k.replace('_', ' ').title()}**: {v}"
                for k, v in self.project_patterns.items()
                if isinstance(v, str)
            )
            if proj_patterns:
                sections.append(f"### Project Conventions\n{proj_patterns}")

        return "\n\n".join(sections)

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        workspace: Path | None = None,
    ) -> ExecutionContext:
        """Create context from executor graph state.

        Args:
            state: ExecutorGraphState dictionary.
            workspace: Optional workspace path override.

        Returns:
            ExecutionContext instance.
        """
        # Extract workspace from state
        ws = workspace
        if ws is None:
            roadmap_path = state.get("roadmap_path", "")
            if roadmap_path:
                ws = Path(roadmap_path).parent
            else:
                ws = Path.cwd()

        # Phase 16.5.11.5: Extract completed tasks from todos
        completed_tasks: list[str] = []
        todos = state.get("todos", [])
        for todo in todos:
            if isinstance(todo, dict) and todo.get("status") == "completed":
                completed_tasks.append(todo.get("title", "Unknown task"))

        # Phase 16.5.11.5: Extract existing files (up to 50)
        existing_files: list[str] = state.get("existing_files", [])
        if not existing_files and ws.exists():
            try:
                # Scan for source files
                for ext in ("*.py", "*.ts", "*.js", "*.tsx", "*.jsx"):
                    for f in ws.rglob(ext):
                        if ".git" not in str(f) and "node_modules" not in str(f):
                            existing_files.append(str(f.relative_to(ws)))
                        if len(existing_files) >= 50:
                            break
                    if len(existing_files) >= 50:
                        break
            except (OSError, ValueError):
                pass

        return cls(
            workspace=ws,
            files_modified=state.get("files_modified", []),
            relevant_files=[],  # Could be populated from context
            project_type="python",  # Default, could detect
            summary=state.get("context", ""),
            run_memory=state.get("run_memory", {}),
            completed_tasks=completed_tasks,
            existing_files=existing_files[:50],  # Limit to 50
            project_patterns=state.get("project_patterns", {}),
            source_file_previews=state.get("source_file_previews", {}),
        )


# =============================================================================
# Subagent Result
# =============================================================================


@dataclass
class SubAgentResult:
    """Result from subagent execution.

    Attributes:
        success: Whether the execution succeeded.
        output: Raw output from the agent.
        files_modified: Files that were modified.
        files_created: Files that were created.
        error: Error message if execution failed.
        metrics: Execution metrics (tokens, time, etc.).
        review_comments: For ReviewerAgent - review comments.
        verdict: For ReviewerAgent - APPROVE or REQUEST_CHANGES.
        tests_run: For TesterAgent - number of tests run.
        tests_passed: For TesterAgent - number of tests passed.
        test_output: For TesterAgent - raw test output.
    """

    success: bool
    output: str = ""
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # ReviewerAgent specific
    review_comments: list[str] = field(default_factory=list)
    verdict: str | None = None

    # TesterAgent specific
    tests_run: int = 0
    tests_passed: int = 0
    test_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "error": self.error,
            "metrics": self.metrics,
            "review_comments": self.review_comments,
            "verdict": self.verdict,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "test_output": self.test_output,
        }

    @property
    def tests_failed(self) -> int:
        """Get number of failed tests."""
        return self.tests_run - self.tests_passed


# =============================================================================
# SubAgent Base Class
# =============================================================================


class SubAgent(ABC):
    """Abstract base class for specialized subagents.

    Subagents are specialized agents optimized for specific tasks.
    Each subagent has:
    - A name and description
    - A configured model (can vary by task complexity)
    - A system prompt tailored to its role
    - Tools appropriate for its function

    Subclasses must implement:
    - `_get_tools()`: Return tools for this agent
    - `execute()`: Execute a task with this agent
    """

    # Class attributes (override in subclasses)
    name: str = "SubAgent"
    description: str = "A specialized subagent"
    model: str = "claude-sonnet-4-20250514"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, model: str | None = None) -> None:
        """Initialize the subagent.

        Args:
            model: Optional model override.
        """
        self._model = model or self.model
        self._agent: Agent | None = None

    @property
    def agent(self) -> Agent:
        """Get or create the underlying agent.

        Lazily creates the agent on first access.
        """
        if self._agent is None:
            from ai_infra.llm.agent import Agent

            self._agent = Agent(
                model_name=self._model,
                tools=self._get_tools(),
                system=self.system_prompt,
            )
        return self._agent

    @abstractmethod
    def _get_tools(self) -> list[Any]:
        """Get tools for this agent.

        Returns:
            List of tools (functions, LangChain tools, etc.)
        """
        pass

    @abstractmethod
    async def execute(
        self,
        task: TodoItem,
        context: ExecutionContext,
    ) -> SubAgentResult:
        """Execute a task with this agent.

        Args:
            task: The task to execute.
            context: Execution context with workspace info.

        Returns:
            SubAgentResult with execution outcome.
        """
        pass

    def __repr__(self) -> str:
        """Get string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', model='{self._model}')"


# =============================================================================
# Default Prompt Templates
# =============================================================================


CODER_SYSTEM_PROMPT = """You are an expert software engineer.

Your job is to write clean, well-tested code that:
1. Follows project conventions
2. Has proper error handling
3. Is well-documented
4. Passes all tests

Available tools:
- read_file: Read existing code
- write_file: Create new files
- edit_file: Modify existing files
- run_shell: Run commands (build, test, etc.)

Always:
1. Read existing code to understand patterns
2. Write code that matches the style
3. Run tests after making changes
"""

REVIEWER_SYSTEM_PROMPT = """You are a senior code reviewer.

Your job is to review code changes and identify:
1. Bugs and logic errors
2. Security vulnerabilities
3. Performance issues
4. Style violations
5. Missing tests

Be constructive but thorough. Provide specific line-by-line feedback.

Output your review in this format:
1. Summary of changes
2. Issues found (with severity: critical/major/minor)
3. Suggestions for improvement
4. Verdict: APPROVE or REQUEST_CHANGES
"""

TESTER_SYSTEM_PROMPT = """You are a QA engineer.

Your job is to:
1. Discover how to run tests in this project
2. Run tests and interpret results
3. Identify flaky tests
4. Report pass/fail status with details

Use run_shell to execute test commands.

For each project type, use appropriate commands:
- Python: pytest, python -m pytest
- Node.js: npm test, jest
- Go: go test ./...
- Rust: cargo test
"""

DEBUGGER_SYSTEM_PROMPT = """You are an expert debugger.

Your job is to:
1. Analyze test failures and error messages
2. Identify root causes
3. Propose and implement fixes
4. Verify fixes work

Approach:
1. Read the error carefully
2. Find the relevant code
3. Understand why it fails
4. Fix the underlying issue (not symptoms)
5. Run tests to verify
"""

RESEARCHER_SYSTEM_PROMPT = """You are a technical researcher.

Your job is to:
1. Search for relevant documentation
2. Find example implementations
3. Research best practices
4. Summarize findings concisely

Use web_search and lookup_docs to find information.
Always cite sources and provide links when available.
"""

TEST_WRITER_SYSTEM_PROMPT = """You are an expert test engineer.

Your job is to CREATE comprehensive test files that:
1. Cover all public functions and methods
2. Test edge cases (empty input, None, type errors)
3. Test error handling paths
4. Use descriptive test names (test_<function>_<scenario>)
5. Include docstrings explaining what each test verifies

Test writing principles:
- Arrange-Act-Assert pattern
- One assertion per test when possible
- Test behavior, not implementation
- Mock external dependencies
- Use fixtures for shared setup

For Python projects:
- Use pytest conventions
- Use pytest.raises for exception testing
- Use @pytest.fixture for setup
- Use parametrize for multiple inputs

Available tools:
- read_file: Read source code to understand what to test
- write_file: Create test files
- run_shell: Run tests to verify they pass

Always:
1. Read the source file to understand the API
2. Identify all testable behaviors
3. Write comprehensive tests
4. Run tests to verify they pass
"""
