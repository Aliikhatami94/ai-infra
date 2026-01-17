"""Context carryover for persisting session context across runs (Phase 5.3).

This module provides infrastructure for remembering context across executor
sessions, enabling agents to maintain project knowledge without re-learning.

Key capabilities:
- SessionSummarizer: Generate summaries of completed sessions
- ContextStorage: Persist and retrieve context per workspace
- ArchitectureTracker: Track project structure understanding
- ContextLoader: Load previous context at session start

Example:
    ```python
    from pathlib import Path
    from ai_infra.executor.context_carryover import (
        ContextStorage,
        SessionSummary,
        ArchitectureTracker,
        load_session_context,
    )

    # At session end, save summary
    storage = ContextStorage()
    summary = SessionSummary(
        session_id="session-123",
        workspace=str(Path.cwd()),
        tasks_completed=["Add auth", "Add tests"],
        project_type="FastAPI API",
        key_patterns=["Use SQLAlchemy for DB"],
        key_decisions=["JWT for auth"],
    )
    storage.save_summary(summary)

    # At next session start, load context
    context_prompt = load_session_context(Path.cwd())
    # Agent now has context from previous session
    ```
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SessionSummary:
    """Summary of an execution session.

    Captures what was done, learned, and what the next session should know.

    Attributes:
        session_id: Unique identifier for the session.
        workspace: Absolute path to the workspace.
        started_at: When the session started.
        completed_at: When the session completed.
        tasks_completed: Titles of completed tasks.
        files_modified: Files that were modified.
        project_type: Detected project type (e.g., "FastAPI API with SQLAlchemy").
        key_patterns: Patterns established during the session.
        key_decisions: Important decisions made and why.
        warnings: Things to watch out for.
        continuation_hint: Suggestions for the next session.
    """

    session_id: str
    workspace: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # What was done
    tasks_completed: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)

    # What was learned
    project_type: str = ""
    key_patterns: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # For next session
    continuation_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "tasks_completed": self.tasks_completed,
            "files_modified": self.files_modified[:50],  # Limit stored files
            "project_type": self.project_type,
            "key_patterns": self.key_patterns[:20],  # Limit patterns
            "key_decisions": self.key_decisions[:20],  # Limit decisions
            "warnings": self.warnings[:10],  # Limit warnings
            "continuation_hint": self.continuation_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSummary:
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            workspace=data["workspace"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            tasks_completed=data.get("tasks_completed", []),
            files_modified=data.get("files_modified", []),
            project_type=data.get("project_type", ""),
            key_patterns=data.get("key_patterns", []),
            key_decisions=data.get("key_decisions", []),
            warnings=data.get("warnings", []),
            continuation_hint=data.get("continuation_hint", ""),
        )

    def to_context_prompt(self) -> str:
        """Generate context prompt for next session.

        Returns:
            Formatted markdown string for injection into agent prompt.
        """
        sections = []

        sections.append("## Previous Session Context")
        sections.append("")

        if self.project_type:
            sections.append(f"**Project Type**: {self.project_type}")
            sections.append("")

        if self.tasks_completed:
            sections.append("**Tasks Completed Previously**:")
            for task in self.tasks_completed[:10]:
                sections.append(f"- {task}")
            if len(self.tasks_completed) > 10:
                sections.append(f"- ... and {len(self.tasks_completed) - 10} more")
            sections.append("")

        if self.key_patterns:
            sections.append("**Established Patterns**:")
            for pattern in self.key_patterns:
                sections.append(f"- {pattern}")
            sections.append("")

        if self.key_decisions:
            sections.append("**Important Decisions**:")
            for decision in self.key_decisions:
                sections.append(f"- {decision}")
            sections.append("")

        if self.warnings:
            sections.append("**Warnings**:")
            for warning in self.warnings:
                sections.append(f"- {warning}")
            sections.append("")

        if self.continuation_hint:
            sections.append(f"**Hint**: {self.continuation_hint}")
            sections.append("")

        return "\n".join(sections)


@dataclass
class ProjectArchitecture:
    """Understanding of project architecture.

    Captures structural patterns and conventions discovered in the project.

    Attributes:
        src_layout: Source layout style ("flat", "src/", "packages/").
        test_layout: Test layout style ("tests/", "same_as_src", "test/").
        config_location: Config location ("root", "config/", ".config/").
        entry_points: Main entry point files.
        core_modules: Most important modules.
        utilities: Helper/utility modules.
        naming_convention: Naming convention ("snake_case", "camelCase").
        import_style: Import style ("absolute", "relative", "mixed").
        docstring_style: Docstring style ("google", "numpy", "sphinx", "none").
        key_dependencies: Key external dependencies.
        internal_imports: How modules import each other.
    """

    # Structure
    src_layout: str = "unknown"
    test_layout: str = "unknown"
    config_location: str = "unknown"

    # Key components
    entry_points: list[str] = field(default_factory=list)
    core_modules: list[str] = field(default_factory=list)
    utilities: list[str] = field(default_factory=list)

    # Patterns
    naming_convention: str = "unknown"
    import_style: str = "unknown"
    docstring_style: str = "unknown"

    # Dependencies
    key_dependencies: list[str] = field(default_factory=list)
    internal_imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectArchitecture:
        """Create from dictionary."""
        return cls(**data)

    def to_context_prompt(self) -> str:
        """Generate context prompt section for architecture.

        Returns:
            Formatted markdown for injection into prompts.
        """
        sections = []
        sections.append("## Project Architecture")
        sections.append("")

        # Structure
        sections.append("**Structure**:")
        sections.append(f"- Source layout: {self.src_layout}")
        sections.append(f"- Test layout: {self.test_layout}")
        sections.append(f"- Config location: {self.config_location}")
        sections.append("")

        # Key components
        if self.entry_points:
            sections.append(f"**Entry Points**: {', '.join(self.entry_points)}")
        if self.core_modules:
            sections.append(f"**Core Modules**: {', '.join(self.core_modules[:5])}")
        if self.utilities:
            sections.append(f"**Utilities**: {', '.join(self.utilities[:5])}")
        sections.append("")

        # Patterns
        sections.append("**Conventions**:")
        sections.append(f"- Naming: {self.naming_convention}")
        sections.append(f"- Imports: {self.import_style}")
        sections.append(f"- Docstrings: {self.docstring_style}")
        sections.append("")

        # Dependencies
        if self.key_dependencies:
            sections.append(f"**Key Dependencies**: {', '.join(self.key_dependencies[:10])}")
            sections.append("")

        return "\n".join(sections)


# =============================================================================
# Context Storage
# =============================================================================


class ContextStorage:
    """Store and retrieve context per workspace.

    Persists session summaries and project architecture to disk,
    organized by workspace hash for isolation.

    Attributes:
        base_path: Base directory for context storage.
    """

    DEFAULT_PATH = Path.home() / ".ai-infra" / "context"
    MAX_HISTORY = 20

    def __init__(self, base_path: Path | str | None = None) -> None:
        """Initialize context storage.

        Args:
            base_path: Base directory for storage (None for default).
        """
        self.base_path = Path(base_path) if base_path else self.DEFAULT_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _workspace_path(self, workspace: Path | str) -> Path:
        """Get storage path for a workspace.

        Uses hash of workspace path for directory name.

        Args:
            workspace: Workspace path.

        Returns:
            Path to workspace storage directory.
        """
        workspace_str = str(workspace)
        workspace_hash = hashlib.sha256(workspace_str.encode()).hexdigest()[:12]
        return self.base_path / workspace_hash

    def save_summary(self, summary: SessionSummary) -> Path:
        """Save a session summary.

        Saves to latest.json and appends to history.jsonl.

        Args:
            summary: Session summary to save.

        Returns:
            Path to the saved latest.json file.
        """
        path = self._workspace_path(summary.workspace)
        path.mkdir(parents=True, exist_ok=True)

        # Save latest summary
        latest_path = path / "latest.json"
        latest_path.write_text(
            json.dumps(summary.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Append to history
        history_path = path / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary.to_dict()) + "\n")

        # Prune history if too long
        self._prune_history(history_path)

        logger.debug(f"Saved session summary for {summary.workspace}")
        return latest_path

    def _prune_history(self, history_path: Path) -> None:
        """Prune history file to max entries.

        Args:
            history_path: Path to history.jsonl file.
        """
        if not history_path.exists():
            return

        lines = history_path.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) > self.MAX_HISTORY:
            # Keep only the latest entries
            lines = lines[-self.MAX_HISTORY :]
            history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def load_summary(self, workspace: Path | str) -> SessionSummary | None:
        """Load most recent session summary for workspace.

        Args:
            workspace: Workspace path.

        Returns:
            SessionSummary or None if not found.
        """
        path = self._workspace_path(workspace)
        latest_path = path / "latest.json"

        if not latest_path.exists():
            return None

        try:
            data = json.loads(latest_path.read_text(encoding="utf-8"))
            return SessionSummary.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load session summary: {e}")
            return None

    def load_history(
        self,
        workspace: Path | str,
        limit: int = 10,
    ) -> list[SessionSummary]:
        """Load session history for workspace.

        Args:
            workspace: Workspace path.
            limit: Maximum entries to return.

        Returns:
            List of SessionSummary, most recent first.
        """
        path = self._workspace_path(workspace)
        history_path = path / "history.jsonl"

        if not history_path.exists():
            return []

        summaries = []
        try:
            lines = history_path.read_text(encoding="utf-8").strip().split("\n")
            for line in reversed(lines[-limit:]):
                if line.strip():
                    data = json.loads(line)
                    summaries.append(SessionSummary.from_dict(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load session history: {e}")

        return summaries

    def save_architecture(
        self,
        workspace: Path | str,
        architecture: ProjectArchitecture,
    ) -> Path:
        """Save project architecture.

        Args:
            workspace: Workspace path.
            architecture: Architecture to save.

        Returns:
            Path to the saved file.
        """
        path = self._workspace_path(workspace)
        path.mkdir(parents=True, exist_ok=True)

        arch_path = path / "architecture.json"
        arch_path.write_text(
            json.dumps(architecture.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.debug(f"Saved architecture for {workspace}")
        return arch_path

    def load_architecture(
        self,
        workspace: Path | str,
    ) -> ProjectArchitecture | None:
        """Load project architecture.

        Args:
            workspace: Workspace path.

        Returns:
            ProjectArchitecture or None if not found.
        """
        path = self._workspace_path(workspace)
        arch_path = path / "architecture.json"

        if not arch_path.exists():
            return None

        try:
            data = json.loads(arch_path.read_text(encoding="utf-8"))
            return ProjectArchitecture.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load architecture: {e}")
            return None

    def clear(self, workspace: Path | str) -> bool:
        """Clear all context for a workspace.

        Args:
            workspace: Workspace path.

        Returns:
            True if cleared, False if nothing to clear.
        """
        path = self._workspace_path(workspace)
        if not path.exists():
            return False

        import shutil

        shutil.rmtree(path)
        return True


# =============================================================================
# Architecture Tracker
# =============================================================================


class ArchitectureTracker:
    """Track and update project architecture understanding.

    Analyzes project structure to build an understanding of
    how the project is organized.
    """

    # Common source layouts
    SRC_LAYOUTS = {
        "src/": ["src/"],
        "lib/": ["lib/"],
        "packages/": ["packages/"],
        "app/": ["app/"],
        "flat": [],  # No dedicated source directory
    }

    # Common test layouts
    TEST_LAYOUTS = {
        "tests/": ["tests/", "test/"],
        "spec/": ["spec/", "specs/"],
        "__tests__/": ["__tests__/"],
        "same_as_src": [],  # Tests alongside source
    }

    # Common entry points
    ENTRY_POINTS = [
        "main.py",
        "app.py",
        "__main__.py",
        "index.py",
        "cli.py",
        "run.py",
        "server.py",
        "index.ts",
        "index.js",
        "main.ts",
        "main.js",
        "app.ts",
        "app.js",
    ]

    # Common config files
    CONFIG_FILES = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
    ]

    def __init__(self, storage: ContextStorage | None = None) -> None:
        """Initialize the tracker.

        Args:
            storage: Context storage for persistence (None creates default).
        """
        self.storage = storage or ContextStorage()

    def analyze(self, workspace: Path) -> ProjectArchitecture:
        """Analyze project architecture.

        Scans the workspace to detect project structure,
        conventions, and key components.

        Args:
            workspace: Path to workspace root.

        Returns:
            Detected ProjectArchitecture.
        """
        arch = ProjectArchitecture()

        # Detect source layout
        arch.src_layout = self._detect_src_layout(workspace)

        # Detect test layout
        arch.test_layout = self._detect_test_layout(workspace)

        # Detect config location
        arch.config_location = self._detect_config_location(workspace)

        # Find entry points
        arch.entry_points = self._find_entry_points(workspace)

        # Find core modules
        arch.core_modules = self._find_core_modules(workspace)

        # Detect naming convention
        arch.naming_convention = self._detect_naming_convention(workspace)

        # Detect import style
        arch.import_style = self._detect_import_style(workspace)

        # Detect docstring style
        arch.docstring_style = self._detect_docstring_style(workspace)

        # Detect key dependencies
        arch.key_dependencies = self._detect_dependencies(workspace)

        return arch

    def _detect_src_layout(self, workspace: Path) -> str:
        """Detect source layout style."""
        for layout, dirs in self.SRC_LAYOUTS.items():
            for d in dirs:
                if (workspace / d).is_dir():
                    return layout
        return "flat"

    def _detect_test_layout(self, workspace: Path) -> str:
        """Detect test layout style."""
        for layout, dirs in self.TEST_LAYOUTS.items():
            for d in dirs:
                if (workspace / d).is_dir():
                    return layout
        return "unknown"

    def _detect_config_location(self, workspace: Path) -> str:
        """Detect configuration location."""
        # Check for config in root
        for cfg in self.CONFIG_FILES:
            if (workspace / cfg).exists():
                return "root"

        # Check for config directory
        if (workspace / "config").is_dir():
            return "config/"

        if (workspace / ".config").is_dir():
            return ".config/"

        return "unknown"

    def _find_entry_points(self, workspace: Path) -> list[str]:
        """Find entry point files."""
        entry_points = []

        # Check common entry point names
        for ep in self.ENTRY_POINTS:
            # Check root
            if (workspace / ep).exists():
                entry_points.append(ep)
            # Check src/
            if (workspace / "src" / ep).exists():
                entry_points.append(f"src/{ep}")
            # Check app/
            if (workspace / "app" / ep).exists():
                entry_points.append(f"app/{ep}")

        return entry_points[:5]  # Limit to 5

    def _find_core_modules(self, workspace: Path) -> list[str]:
        """Find core modules by directory structure."""
        core = []

        # Look in src/ or root
        src_dir = workspace / "src"
        if not src_dir.is_dir():
            src_dir = workspace

        # Find Python packages (directories with __init__.py)
        for item in src_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                core.append(item.name)

        return core[:10]

    def _detect_naming_convention(self, workspace: Path) -> str:
        """Detect naming convention from file names."""
        python_files = list(workspace.rglob("*.py"))[:50]

        snake_count = 0
        camel_count = 0

        for f in python_files:
            name = f.stem
            if "_" in name:
                snake_count += 1
            elif name != name.lower() and name != name.upper():
                # Has mixed case but no underscores
                if name[0].islower():
                    camel_count += 1

        if snake_count > camel_count:
            return "snake_case"
        elif camel_count > snake_count:
            return "camelCase"
        return "snake_case"  # Default for Python

    def _detect_import_style(self, workspace: Path) -> str:
        """Detect import style from Python files."""
        python_files = list(workspace.rglob("*.py"))[:20]

        absolute_count = 0
        relative_count = 0

        for f in python_files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                absolute_count += len(re.findall(r"^from [a-zA-Z]", content, re.MULTILINE))
                relative_count += len(re.findall(r"^from \.", content, re.MULTILINE))
            except OSError:
                continue

        if absolute_count > relative_count * 2:
            return "absolute"
        elif relative_count > absolute_count * 2:
            return "relative"
        return "mixed"

    def _detect_docstring_style(self, workspace: Path) -> str:
        """Detect docstring style from Python files."""
        python_files = list(workspace.rglob("*.py"))[:20]

        for f in python_files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")

                # Google style: Args:, Returns:
                if "Args:" in content or "Returns:" in content:
                    return "google"

                # NumPy style: Parameters\n    ----------
                if "Parameters\n" in content and "---" in content:
                    return "numpy"

                # Sphinx style: :param, :returns:
                if ":param " in content or ":returns:" in content:
                    return "sphinx"
            except OSError:
                continue

        return "unknown"

    def _detect_dependencies(self, workspace: Path) -> list[str]:
        """Detect key dependencies from project files."""
        deps = set()

        # Check pyproject.toml
        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                # Simple regex to find dependencies
                matches = re.findall(r"^([a-zA-Z0-9_-]+)\s*=", content, re.MULTILINE)
                deps.update(matches[:20])
            except OSError:
                pass

        # Check requirements.txt
        requirements = workspace / "requirements.txt"
        if requirements.exists():
            try:
                for line in requirements.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Extract package name
                        match = re.match(r"^([a-zA-Z0-9_-]+)", line)
                        if match:
                            deps.add(match.group(1))
            except OSError:
                pass

        # Check package.json
        package_json = workspace / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8"))
                deps.update(data.get("dependencies", {}).keys())
                deps.update(data.get("devDependencies", {}).keys())
            except (OSError, json.JSONDecodeError):
                pass

        return sorted(deps)[:20]

    def update(
        self,
        architecture: ProjectArchitecture,
        changes: list[str],
    ) -> ProjectArchitecture:
        """Update architecture based on file changes.

        Args:
            architecture: Current architecture.
            changes: List of changed file paths.

        Returns:
            Updated architecture.
        """
        # Update entry points if main files changed
        for change in changes:
            basename = Path(change).name
            if basename in self.ENTRY_POINTS:
                if change not in architecture.entry_points:
                    architecture.entry_points.append(change)

        # Deduplicate and limit
        architecture.entry_points = list(set(architecture.entry_points))[:5]
        architecture.core_modules = list(set(architecture.core_modules))[:10]

        return architecture

    def save(
        self,
        workspace: Path,
        architecture: ProjectArchitecture,
    ) -> Path:
        """Save architecture to storage.

        Args:
            workspace: Workspace path.
            architecture: Architecture to save.

        Returns:
            Path to saved file.
        """
        return self.storage.save_architecture(workspace, architecture)

    def load(self, workspace: Path) -> ProjectArchitecture | None:
        """Load architecture from storage.

        Args:
            workspace: Workspace path.

        Returns:
            ProjectArchitecture or None.
        """
        return self.storage.load_architecture(workspace)


# =============================================================================
# Context Loading
# =============================================================================


def load_session_context(
    workspace: Path | str,
    storage: ContextStorage | None = None,
) -> str:
    """Load previous session context for injection into prompts.

    This is the main entry point for context carryover at session start.

    Args:
        workspace: Workspace path.
        storage: Context storage (None creates default).

    Returns:
        Context prompt string (empty if no previous context).

    Example:
        ```python
        context = load_session_context(Path.cwd())
        if context:
            prompt = f"{context}\\n\\n{task_prompt}"
        ```
    """
    storage = storage or ContextStorage()
    workspace = Path(workspace)

    sections = []

    # Load previous session summary
    summary = storage.load_summary(workspace)
    if summary:
        sections.append(summary.to_context_prompt())

    # Load architecture
    architecture = storage.load_architecture(workspace)
    if architecture:
        sections.append(architecture.to_context_prompt())

    return "\n".join(sections)


def save_session_context(
    summary: SessionSummary,
    architecture: ProjectArchitecture | None = None,
    storage: ContextStorage | None = None,
) -> None:
    """Save session context for future sessions.

    Args:
        summary: Session summary to save.
        architecture: Optional architecture to save.
        storage: Context storage (None creates default).

    Example:
        ```python
        summary = SessionSummary(
            session_id="sess-123",
            workspace=str(Path.cwd()),
            tasks_completed=["Task 1", "Task 2"],
            project_type="FastAPI API",
        )
        save_session_context(summary)
        ```
    """
    storage = storage or ContextStorage()

    storage.save_summary(summary)

    if architecture:
        storage.save_architecture(summary.workspace, architecture)


# =============================================================================
# Context Summarizer (Phase 9.2)
# =============================================================================


def _count_tokens_estimate(text: str) -> int:
    """Estimate token count for a string.

    Uses the standard heuristic of ~4 characters per token.
    This is intentionally conservative (overestimates) to stay within limits.

    Args:
        text: Text to count tokens for.

    Returns:
        Estimated token count.
    """
    return (len(text) + 3) // 4


def _extract_keywords_from_text(text: str) -> set[str]:
    """Extract significant keywords from text for relevance matching.

    Args:
        text: Text to extract keywords from.

    Returns:
        Set of lowercase keywords.
    """
    # Remove punctuation and split
    words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())
    # Filter out common stop words and short words
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "it",
    }
    return {w for w in words if len(w) > 2 and w not in stop_words}


@dataclass
class ContextSummarizer:
    """Summarize context from previous tasks to fit token limits.

    Provides intelligent summarization of task outcomes for injection into
    subsequent task prompts, with relevance filtering and token budgeting.

    Phase 9.2 implementation for smart context carryover.

    Attributes:
        max_tokens: Maximum tokens for generated summary.
        max_tasks: Maximum number of tasks to include in summary.
        relevance_threshold: Minimum keyword overlap for relevance (0.0-1.0).

    Example:
        ```python
        from ai_infra.executor.context_carryover import ContextSummarizer
        from ai_infra.executor.todolist import TodoItem

        summarizer = ContextSummarizer(max_tokens=1000)

        previous_tasks = [
            {"title": "Create User model", "files": ["models.py"], "summary": "Added User class"},
            {"title": "Add tests", "files": ["test_models.py"], "summary": "Added pytest tests"},
        ]

        current_task = TodoItem(
            id=3,
            title="Create UserService that uses User model",
            description="Implement service layer for user operations",
        )

        summary = summarizer.summarize_for_task(previous_tasks, current_task)
        # Returns formatted markdown with relevant previous task context
        ```
    """

    max_tokens: int = 1000
    max_tasks: int = 3
    relevance_threshold: float = 0.1

    def summarize_for_task(
        self,
        previous_tasks: list[dict[str, Any]],
        current_task: Any,
        max_tokens: int | None = None,
    ) -> str:
        """Create relevant summary for current task.

        Filters previous tasks by relevance to the current task and formats
        them into a token-efficient summary for prompt injection.

        Args:
            previous_tasks: List of task dictionaries with title, files, summary keys.
            current_task: Current TodoItem or object with title/description attributes.
            max_tokens: Optional override for max tokens (uses self.max_tokens if None).

        Returns:
            Formatted markdown summary, or empty string if no relevant context.
        """
        if not previous_tasks:
            return ""

        budget = max_tokens if max_tokens is not None else self.max_tokens

        # Get relevant tasks
        relevant = self._filter_relevant_tasks(previous_tasks, current_task)

        if not relevant:
            return ""

        # Format as summary within token budget
        return self._format_summary(relevant, budget)

    def _filter_relevant_tasks(
        self,
        previous_tasks: list[dict[str, Any]],
        current_task: Any,
    ) -> list[dict[str, Any]]:
        """Filter tasks by relevance to current task.

        Uses keyword overlap to determine relevance. Tasks are scored by
        how many keywords they share with the current task.

        Args:
            previous_tasks: List of task dictionaries.
            current_task: Current task object.

        Returns:
            List of relevant task dictionaries, sorted by relevance.
        """
        # Extract keywords from current task
        current_title = getattr(current_task, "title", "") or ""
        current_desc = getattr(current_task, "description", "") or ""
        current_files = getattr(current_task, "file_hints", []) or []

        current_text = f"{current_title} {current_desc} {' '.join(current_files)}"
        current_keywords = _extract_keywords_from_text(current_text)

        if not current_keywords:
            # No keywords to match - return most recent tasks
            return list(reversed(previous_tasks[-self.max_tasks :]))

        # Score each previous task
        scored_tasks: list[tuple[float, dict[str, Any]]] = []

        for task in previous_tasks:
            task_title = task.get("title", "")
            task_summary = task.get("summary", "")
            task_files = task.get("files", [])
            task_decisions = task.get("key_decisions", [])

            # Build task text for keyword extraction
            task_text = f"{task_title} {task_summary} {' '.join(task_files)}"
            if isinstance(task_decisions, list):
                task_text += " " + " ".join(task_decisions)

            task_keywords = _extract_keywords_from_text(task_text)

            if not task_keywords:
                continue

            # Calculate relevance as Jaccard-like overlap
            overlap = len(current_keywords & task_keywords)
            union = len(current_keywords | task_keywords)
            relevance = overlap / union if union > 0 else 0.0

            # Also check for file overlap (strong signal)
            current_file_stems = {Path(f).stem.lower() for f in current_files if f}
            task_file_stems = {Path(f).stem.lower() for f in task_files if f}
            if current_file_stems & task_file_stems:
                relevance += 0.3  # Boost for file overlap

            if relevance >= self.relevance_threshold:
                scored_tasks.append((relevance, task))

        # Sort by relevance (highest first) and take top tasks
        scored_tasks.sort(key=lambda x: x[0], reverse=True)
        return [task for _, task in scored_tasks[: self.max_tasks]]

    def _format_summary(
        self,
        relevant_tasks: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """Format relevant tasks as a summary within token budget.

        Args:
            relevant_tasks: List of relevant task dictionaries.
            max_tokens: Maximum tokens for the summary.

        Returns:
            Formatted markdown summary.
        """
        if not relevant_tasks:
            return ""

        # Reserve tokens for header
        header = "## Context from Previous Tasks\n"
        header_tokens = _count_tokens_estimate(header)
        remaining_budget = max_tokens - header_tokens

        if remaining_budget <= 0:
            return ""

        summary_parts = [header]
        tokens_used = header_tokens

        for task in relevant_tasks:
            task_section = self._format_task_section(task)
            section_tokens = _count_tokens_estimate(task_section)

            if tokens_used + section_tokens > max_tokens:
                # Try compact format
                task_section = self._format_task_compact(task)
                section_tokens = _count_tokens_estimate(task_section)

                if tokens_used + section_tokens > max_tokens:
                    # Still too big, skip this task
                    continue

            summary_parts.append(task_section)
            tokens_used += section_tokens

        if len(summary_parts) == 1:
            # Only header, nothing else fit
            return ""

        return "\n".join(summary_parts)

    def _format_task_section(self, task: dict[str, Any]) -> str:
        """Format a single task as a markdown section.

        Args:
            task: Task dictionary.

        Returns:
            Formatted markdown section.
        """
        title = task.get("title", "Unknown Task")
        files = task.get("files", [])
        summary = task.get("summary", "Completed successfully")
        key_decisions = task.get("key_decisions", [])

        # Format files (limit to 3)
        if files:
            files_display = ", ".join(str(f) for f in files[:3])
            if len(files) > 3:
                files_display += f" (+{len(files) - 3} more)"
        else:
            files_display = "none"

        lines = [
            f"### {title} (completed)",
            f"- Files modified: {files_display}",
            f"- Key outcome: {summary}",
        ]

        # Add key decisions if available (limit to 2)
        if key_decisions:
            decisions_display = "; ".join(key_decisions[:2])
            if len(key_decisions) > 2:
                decisions_display += f" (+{len(key_decisions) - 2} more)"
            lines.append(f"- Decisions: {decisions_display}")

        lines.append("")  # Blank line after section
        return "\n".join(lines)

    def _format_task_compact(self, task: dict[str, Any]) -> str:
        """Format a task in compact form for tight token budgets.

        Args:
            task: Task dictionary.

        Returns:
            Compact formatted string.
        """
        title = task.get("title", "Task")
        files = task.get("files", [])
        summary = task.get("summary", "Done")

        # Very compact: just title and key file
        files_hint = Path(files[0]).name if files else "no files"
        return f"- **{title}**: {summary} ({files_hint})\n"

    def is_relevant(
        self,
        previous_task: dict[str, Any],
        current_task: Any,
    ) -> bool:
        """Check if a previous task is relevant to the current task.

        Convenience method for checking single task relevance.

        Args:
            previous_task: Previous task dictionary.
            current_task: Current task object.

        Returns:
            True if the previous task is relevant.
        """
        relevant = self._filter_relevant_tasks([previous_task], current_task)
        return len(relevant) > 0


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Models
    "SessionSummary",
    "ProjectArchitecture",
    # Storage
    "ContextStorage",
    # Tracker
    "ArchitectureTracker",
    # Summarizer
    "ContextSummarizer",
    # Functions
    "load_session_context",
    "save_session_context",
]
