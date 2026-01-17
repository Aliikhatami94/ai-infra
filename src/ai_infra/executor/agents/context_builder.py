"""Rich context builder for subagent execution.

Phase 16.5.12 of EXECUTOR_5.md: Build comprehensive context from
completed tasks, project patterns, and workspace files to enable
subagents to produce higher quality output.

This module provides:
- SubagentContext: Dataclass holding rich context information
- SubagentContextBuilder: Builds context from workspace and task history

Example:
    ```python
    from ai_infra.executor.agents.context_builder import SubagentContextBuilder
    from ai_infra.executor.todolist import TodoItem
    from pathlib import Path

    builder = SubagentContextBuilder()
    context = builder.build(
        task=TodoItem(id=1, title="Create tests for user.py", description=""),
        workspace=Path("/project"),
        completed_tasks=[TodoItem(id=0, title="Create src/user.py", description="")],
    )
    print(context.format_for_prompt())
    ```
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

__all__ = [
    "CodePatterns",
    "SubagentContext",
    "SubagentContextBuilder",
    "ValidationResult",
]

logger = get_logger("executor.agents.context_builder")


# =============================================================================
# Constants
# =============================================================================

# Maximum tokens for context (roughly 4 chars per token)
MAX_CONTEXT_TOKENS = 2000
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4

# File extensions to consider as source code
SOURCE_EXTENSIONS = {".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go", ".java"}

# Files to ignore when scanning workspace
IGNORE_PATTERNS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".egg-info",
    "htmlcov",
    ".coverage",
}

# Maximum files to list
MAX_FILES_TO_LIST = 30

# Maximum lines for file preview
MAX_PREVIEW_LINES = 50


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CodePatterns:
    """Detected code patterns from existing codebase.

    Phase 16.5.12.1.6: Captures coding conventions to help subagents
    match project style.

    Attributes:
        docstring_style: Detected docstring format (google, numpy, sphinx, none).
        import_style: Import organization style (grouped, sorted, unsorted).
        test_framework: Detected test framework (pytest, unittest, jest).
        type_hints: Whether type hints are used (yes, partial, no).
        naming_convention: Variable naming style (snake_case, camelCase).
        indent_style: Indentation style (spaces-4, spaces-2, tabs).
        line_length: Typical max line length.
    """

    docstring_style: str = "unknown"
    import_style: str = "unknown"
    test_framework: str = "pytest"
    type_hints: str = "unknown"
    naming_convention: str = "snake_case"
    indent_style: str = "spaces-4"
    line_length: int = 88

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "docstring_style": self.docstring_style,
            "import_style": self.import_style,
            "test_framework": self.test_framework,
            "type_hints": self.type_hints,
            "naming_convention": self.naming_convention,
            "indent_style": self.indent_style,
            "line_length": self.line_length,
        }


@dataclass
class SubagentContext:
    """Rich context for subagent execution.

    Phase 16.5.12.1.2: Contains all the information a subagent needs
    to produce high-quality output that matches project conventions.

    Attributes:
        task: The current task being executed.
        workspace: Path to the workspace root.
        project_type: Detected project type (python, node, rust, etc.).
        existing_files: List of files in the workspace.
        file_previews: Content previews of relevant files.
        completed_summaries: One-line summaries of completed tasks.
        code_patterns: Detected coding patterns and conventions.
        dependencies: Project dependencies (from pyproject.toml, package.json).
    """

    task: TodoItem
    workspace: Path
    project_type: str = "unknown"
    existing_files: list[str] = field(default_factory=list)
    file_previews: dict[str, str] = field(default_factory=dict)
    completed_summaries: list[str] = field(default_factory=list)
    code_patterns: CodePatterns = field(default_factory=CodePatterns)
    dependencies: list[str] = field(default_factory=list)

    def format_for_prompt(self, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Format context for injection into subagent prompt.

        Phase 16.5.12.2.2: Creates a structured context string that
        fits within the token budget.

        Args:
            max_chars: Maximum characters for the formatted context.

        Returns:
            Formatted context string.
        """
        sections: list[str] = []

        # Project info section
        project_section = f"""## Project Context

**Project Type**: {self.project_type}
**Dependencies**: {", ".join(self.dependencies[:10]) or "none detected"}"""
        sections.append(project_section)

        # Previously completed tasks
        if self.completed_summaries:
            tasks_text = "\n".join(f"- {s}" for s in self.completed_summaries[-5:])
            sections.append(f"## Previously Completed Tasks\n{tasks_text}")

        # Relevant existing files
        if self.existing_files:
            # Prioritize source files
            source_files = [
                f for f in self.existing_files if any(f.endswith(ext) for ext in SOURCE_EXTENSIONS)
            ]
            other_files = [f for f in self.existing_files if f not in source_files]
            display_files = (source_files[:10] + other_files[:5])[:15]
            files_text = "\n".join(f"- {f}" for f in display_files)
            if len(self.existing_files) > 15:
                files_text += f"\n- ... and {len(self.existing_files) - 15} more files"
            sections.append(f"## Relevant Existing Files\n{files_text}")

        # Code patterns
        patterns = self.code_patterns
        patterns_section = f"""## Code Patterns Detected
- Docstring style: {patterns.docstring_style}
- Import style: {patterns.import_style}
- Test framework: {patterns.test_framework}
- Type hints: {patterns.type_hints}
- Naming: {patterns.naming_convention}"""
        sections.append(patterns_section)

        # File previews (budget-aware)
        if self.file_previews:
            preview_texts: list[str] = []
            remaining_chars = max_chars - len("\n\n".join(sections)) - 500

            for filepath, content in self.file_previews.items():
                # Truncate preview if needed
                if len(content) > remaining_chars // len(self.file_previews):
                    lines = content.split("\n")
                    truncated = "\n".join(lines[:30])
                    preview_text = f"### {filepath}\n```python\n{truncated}\n# ... (truncated)\n```"
                else:
                    preview_text = f"### {filepath}\n```python\n{content}\n```"

                preview_texts.append(preview_text)
                remaining_chars -= len(preview_text)

                if remaining_chars < 500:
                    break

            if preview_texts:
                sections.append("## File Previews\n" + "\n\n".join(preview_texts))

        result = "\n\n".join(sections)

        # Final truncation if needed
        if len(result) > max_chars:
            result = result[: max_chars - 50] + "\n\n... (context truncated)"

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_title": self.task.title if self.task else "",
            "workspace": str(self.workspace),
            "project_type": self.project_type,
            "existing_files_count": len(self.existing_files),
            "file_previews_count": len(self.file_previews),
            "completed_tasks_count": len(self.completed_summaries),
            "code_patterns": self.code_patterns.to_dict(),
            "dependencies_count": len(self.dependencies),
        }


@dataclass
class ValidationResult:
    """Result of subagent output validation.

    Phase 16.5.12.3.2: Contains validation status and any issues found.

    Attributes:
        valid: Whether the output passed validation.
        issues: List of issues found.
        score: Quality score from 0 to 1.
    """

    valid: bool = True
    issues: list[str] = field(default_factory=list)
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "issues": self.issues,
            "score": self.score,
        }


# =============================================================================
# Context Builder
# =============================================================================


class SubagentContextBuilder:
    """Builds rich context for subagent execution.

    Phase 16.5.12.1: Analyzes workspace and task history to create
    comprehensive context that helps subagents produce better output.

    Example:
        ```python
        builder = SubagentContextBuilder()
        context = builder.build(
            task=task,
            workspace=Path("/project"),
            completed_tasks=completed,
        )
        prompt = f"{system_prompt}\\n\\n{context.format_for_prompt()}"
        ```
    """

    def build(
        self,
        task: TodoItem,
        workspace: Path,
        completed_tasks: list[TodoItem] | None = None,
        run_memory: dict[str, Any] | None = None,
    ) -> SubagentContext:
        """Build comprehensive context for subagent.

        Phase 16.5.12.1.2: Main entry point for context building.

        Args:
            task: Current task to execute.
            workspace: Workspace root path.
            completed_tasks: Previously completed tasks.
            run_memory: Optional memory from previous runs.

        Returns:
            SubagentContext with rich project information.
        """
        completed_tasks = completed_tasks or []
        run_memory = run_memory or {}

        # Detect project type
        project_type = self._detect_project_type(workspace)

        # List relevant files
        existing_files = self._list_relevant_files(workspace, task)

        # Get file previews for related files
        file_previews = self._get_file_previews(workspace, task, existing_files)

        # Summarize completed tasks
        completed_summaries = self._summarize_completed(completed_tasks)

        # Extract code patterns
        code_patterns = self._extract_patterns(workspace, existing_files)

        # Detect dependencies
        dependencies = self._detect_dependencies(workspace)

        context = SubagentContext(
            task=task,
            workspace=workspace,
            project_type=project_type,
            existing_files=existing_files,
            file_previews=file_previews,
            completed_summaries=completed_summaries,
            code_patterns=code_patterns,
            dependencies=dependencies,
        )

        logger.debug(
            f"Built context: {len(existing_files)} files, "
            f"{len(file_previews)} previews, {len(completed_summaries)} tasks"
        )

        return context

    def _detect_project_type(self, workspace: Path) -> str:
        """Detect project type from workspace files.

        Phase 16.5.12.1.2: Identifies the primary language/framework.

        Args:
            workspace: Workspace root path.

        Returns:
            Project type string (python, node, rust, etc.).
        """
        # Check for Python
        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists():
            return "python"

        # Check for Node.js
        if (workspace / "package.json").exists():
            return "node"

        # Check for Rust
        if (workspace / "Cargo.toml").exists():
            return "rust"

        # Check for Go
        if (workspace / "go.mod").exists():
            return "go"

        # Check by file extensions
        py_files = list(workspace.glob("**/*.py"))
        ts_files = list(workspace.glob("**/*.ts"))
        js_files = list(workspace.glob("**/*.js"))

        if len(py_files) > len(ts_files) + len(js_files):
            return "python"
        if len(ts_files) > 0:
            return "typescript"
        if len(js_files) > 0:
            return "javascript"

        return "unknown"

    def _list_relevant_files(
        self,
        workspace: Path,
        task: TodoItem,
    ) -> list[str]:
        """List files relevant to the current task.

        Phase 16.5.12.1.3: Finds files related to the task by name
        matching and directory proximity.

        Args:
            workspace: Workspace root path.
            task: Current task.

        Returns:
            List of relevant file paths (relative to workspace).
        """
        all_files: list[str] = []

        # Walk workspace and collect files
        try:
            for path in workspace.rglob("*"):
                if path.is_file():
                    # Skip ignored directories
                    if any(ignore in path.parts for ignore in IGNORE_PATTERNS):
                        continue

                    rel_path = str(path.relative_to(workspace))
                    all_files.append(rel_path)

                    if len(all_files) >= MAX_FILES_TO_LIST * 2:
                        break
        except Exception as e:
            logger.warning(f"Error listing files: {e}")
            return []

        # Prioritize files related to task
        task_terms = self._extract_task_terms(task)

        def relevance_score(filepath: str) -> int:
            """Score file relevance to task."""
            score = 0
            filepath_lower = filepath.lower()

            for term in task_terms:
                if term in filepath_lower:
                    score += 10

            # Prioritize source files
            if any(filepath.endswith(ext) for ext in SOURCE_EXTENSIONS):
                score += 5

            # Prioritize src/ and tests/
            if filepath.startswith(("src/", "tests/")):
                score += 3

            return score

        # Sort by relevance and return top files
        sorted_files = sorted(all_files, key=relevance_score, reverse=True)
        return sorted_files[:MAX_FILES_TO_LIST]

    def _extract_task_terms(self, task: TodoItem) -> list[str]:
        """Extract searchable terms from task.

        Args:
            task: The task to analyze.

        Returns:
            List of lowercase terms to search for.
        """
        text = f"{task.title} {task.description or ''}".lower()

        # Extract potential file/module names
        # Match patterns like user.py, test_user, UserService
        patterns = [
            r"\b([a-z_]+\.py)\b",  # file.py
            r"\btest_([a-z_]+)\b",  # test_something
            r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b",  # CamelCase
            r"\b([a-z_]{3,})\b",  # snake_case words
        ]

        terms: set[str] = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match) >= 3:
                    terms.add(match.lower())

        return list(terms)

    def _get_file_previews(
        self,
        workspace: Path,
        task: TodoItem,
        existing_files: list[str],
    ) -> dict[str, str]:
        """Get content previews of relevant files.

        Phase 16.5.12.1.4: Reads first N lines of files related to task.

        Args:
            workspace: Workspace root path.
            task: Current task.
            existing_files: List of files in workspace.

        Returns:
            Dictionary mapping filepath to content preview.
        """
        previews: dict[str, str] = {}
        task_terms = self._extract_task_terms(task)

        # Find most relevant source files
        relevant_files: list[str] = []
        for filepath in existing_files:
            if not any(filepath.endswith(ext) for ext in SOURCE_EXTENSIONS):
                continue

            # Check if file relates to task
            filepath_lower = filepath.lower()
            if any(term in filepath_lower for term in task_terms):
                relevant_files.append(filepath)

        # Read previews (max 5 files)
        for filepath in relevant_files[:5]:
            full_path = workspace / filepath
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                preview = "\n".join(lines[:MAX_PREVIEW_LINES])
                if len(lines) > MAX_PREVIEW_LINES:
                    preview += f"\n# ... ({len(lines) - MAX_PREVIEW_LINES} more lines)"
                previews[filepath] = preview
            except Exception as e:
                logger.debug(f"Could not read {filepath}: {e}")

        return previews

    def _summarize_completed(
        self,
        completed_tasks: list[TodoItem],
    ) -> list[str]:
        """Create one-line summaries of completed tasks.

        Phase 16.5.12.1.5: Summarizes completed work for context.

        Args:
            completed_tasks: List of completed tasks.

        Returns:
            List of summary strings.
        """
        summaries: list[str] = []

        for task in completed_tasks[-10:]:  # Last 10 tasks
            # Create concise summary
            title = task.title
            if len(title) > 80:
                title = title[:77] + "..."
            summaries.append(f"[Done] {title}")

        return summaries

    def _extract_patterns(
        self,
        workspace: Path,
        existing_files: list[str],
    ) -> CodePatterns:
        """Extract coding patterns from existing code.

        Phase 16.5.12.1.6: Analyzes code style and conventions.

        Args:
            workspace: Workspace root path.
            existing_files: List of files to analyze.

        Returns:
            CodePatterns with detected conventions.
        """
        patterns = CodePatterns()

        # Analyze Python files
        py_files = [f for f in existing_files if f.endswith(".py")]
        if not py_files:
            return patterns

        docstring_styles: dict[str, int] = {"google": 0, "numpy": 0, "sphinx": 0, "none": 0}
        type_hint_count = 0
        total_functions = 0

        for filepath in py_files[:10]:  # Analyze up to 10 files
            full_path = workspace / filepath
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")

                # Detect docstring style
                if "Args:" in content and "Returns:" in content:
                    docstring_styles["google"] += 1
                elif "Parameters" in content and "----------" in content:
                    docstring_styles["numpy"] += 1
                elif ":param " in content or ":returns:" in content:
                    docstring_styles["sphinx"] += 1
                else:
                    docstring_styles["none"] += 1

                # Check for type hints
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if node.returns or any(arg.annotation for arg in node.args.args):
                                type_hint_count += 1
                except SyntaxError:
                    pass

            except Exception as e:
                logger.debug(f"Could not analyze {filepath}: {e}")

        # Determine patterns
        if docstring_styles:
            patterns.docstring_style = max(docstring_styles, key=docstring_styles.get)

        if total_functions > 0:
            ratio = type_hint_count / total_functions
            if ratio > 0.8:
                patterns.type_hints = "yes"
            elif ratio > 0.3:
                patterns.type_hints = "partial"
            else:
                patterns.type_hints = "no"

        # Check for test framework
        if any("pytest" in f for f in existing_files) or (workspace / "pytest.ini").exists():
            patterns.test_framework = "pytest"
        elif any("unittest" in f for f in existing_files):
            patterns.test_framework = "unittest"

        # Check pyproject.toml for line length
        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                # Look for line-length setting
                match = re.search(r"line-length\s*=\s*(\d+)", content)
                if match:
                    patterns.line_length = int(match.group(1))
            except Exception:
                pass

        return patterns

    def _detect_dependencies(self, workspace: Path) -> list[str]:
        """Detect project dependencies.

        Phase 16.5.12.1.7: Extracts dependencies from config files.

        Args:
            workspace: Workspace root path.

        Returns:
            List of dependency names.
        """
        dependencies: list[str] = []

        # Check pyproject.toml
        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()

                # Look for dependencies section
                in_deps = False
                for line in content.split("\n"):
                    if "[tool.poetry.dependencies]" in line or "[project.dependencies]" in line:
                        in_deps = True
                        continue
                    if in_deps:
                        if line.startswith("["):
                            break
                        # Extract package name
                        match = re.match(r"^([a-zA-Z0-9_-]+)\s*=", line.strip())
                        if match:
                            pkg = match.group(1)
                            if pkg != "python":
                                dependencies.append(pkg)
            except Exception as e:
                logger.debug(f"Could not parse pyproject.toml: {e}")

        # Check requirements.txt
        requirements = workspace / "requirements.txt"
        if requirements.exists():
            try:
                content = requirements.read_text()
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Extract package name (before ==, >=, etc.)
                        match = re.match(r"^([a-zA-Z0-9_-]+)", line)
                        if match:
                            dependencies.append(match.group(1))
            except Exception as e:
                logger.debug(f"Could not parse requirements.txt: {e}")

        # Check package.json
        package_json = workspace / "package.json"
        if package_json.exists():
            try:
                import json

                content = package_json.read_text()
                data = json.loads(content)
                deps = data.get("dependencies", {})
                dependencies.extend(deps.keys())
            except Exception as e:
                logger.debug(f"Could not parse package.json: {e}")

        return list(set(dependencies))[:20]  # Dedupe and limit
