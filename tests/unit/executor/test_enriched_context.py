"""Tests for enriched subagent context (Phase 16.5.5).

These tests verify that subagents receive rich context including:
- Previous task summaries
- Established code patterns
- File contents cache
- Project type detection
- Session briefs
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from ai_infra.executor.agents.base import ExecutionContext
from ai_infra.executor.nodes.execute import (
    _build_execution_context,
    _build_file_contents_cache,
    _build_session_brief,
    _build_task_summaries,
    _detect_project_type,
    _extract_code_patterns,
    _get_file_preview,
    _get_relevant_files,
)


class TestDetectProjectType:
    """Tests for project type detection."""

    def test_detect_python_pyproject(self, tmp_path: Path) -> None:
        """Detect Python project from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        assert _detect_project_type(tmp_path) == "python"

    def test_detect_python_setup_py(self, tmp_path: Path) -> None:
        """Detect Python project from setup.py."""
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        assert _detect_project_type(tmp_path) == "python"

    def test_detect_python_requirements(self, tmp_path: Path) -> None:
        """Detect Python project from requirements.txt."""
        (tmp_path / "requirements.txt").write_text("requests==2.28.0")
        assert _detect_project_type(tmp_path) == "python"

    def test_detect_node_package_json(self, tmp_path: Path) -> None:
        """Detect Node.js project from package.json."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        assert _detect_project_type(tmp_path) == "node"

    def test_detect_rust_cargo_toml(self, tmp_path: Path) -> None:
        """Detect Rust project from Cargo.toml."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'")
        assert _detect_project_type(tmp_path) == "rust"

    def test_detect_go_mod(self, tmp_path: Path) -> None:
        """Detect Go project from go.mod."""
        (tmp_path / "go.mod").write_text("module example.com/test")
        assert _detect_project_type(tmp_path) == "go"

    def test_detect_unknown(self, tmp_path: Path) -> None:
        """Return unknown for unrecognized projects."""
        assert _detect_project_type(tmp_path) == "unknown"

    def test_python_priority_over_node(self, tmp_path: Path) -> None:
        """Python should be detected first if both exist."""
        (tmp_path / "pyproject.toml").write_text("[project]")
        (tmp_path / "package.json").write_text("{}")
        assert _detect_project_type(tmp_path) == "python"


class TestFilePreview:
    """Tests for file preview function."""

    def test_preview_small_file(self, tmp_path: Path) -> None:
        """Get full content of small file."""
        content = "line1\nline2\nline3"
        file = tmp_path / "test.txt"
        file.write_text(content)

        preview = _get_file_preview(file, max_lines=50)
        assert preview == content

    def test_preview_large_file_truncated(self, tmp_path: Path) -> None:
        """Truncate large files to max_lines."""
        content = "\n".join(f"line{i}" for i in range(100))
        file = tmp_path / "test.txt"
        file.write_text(content)

        preview = _get_file_preview(file, max_lines=10)
        assert preview is not None
        lines = preview.split("\n")
        assert len(lines) == 11  # 10 lines + truncation message
        assert "more lines" in lines[-1]

    def test_preview_nonexistent_file(self, tmp_path: Path) -> None:
        """Return None for nonexistent files."""
        file = tmp_path / "nonexistent.txt"
        assert _get_file_preview(file) is None

    def test_preview_directory(self, tmp_path: Path) -> None:
        """Return None for directories."""
        assert _get_file_preview(tmp_path) is None


class TestExtractCodePatterns:
    """Tests for code pattern extraction."""

    def test_detect_google_docstring(self, tmp_path: Path) -> None:
        """Detect Google-style docstrings."""
        content = dedent('''
            def foo():
                """Do something.

                Args:
                    x: Input value.

                Returns:
                    Result value.
                """
                pass
        ''')
        (tmp_path / "test.py").write_text(content)

        patterns = _extract_code_patterns(tmp_path, ["test.py"])
        assert patterns.get("docstring_style") == "google"

    def test_detect_sphinx_docstring(self, tmp_path: Path) -> None:
        """Detect Sphinx-style docstrings."""
        content = dedent('''
            def foo():
                """Do something.

                :param x: Input value.
                :returns: Result value.
                """
                pass
        ''')
        (tmp_path / "test.py").write_text(content)

        patterns = _extract_code_patterns(tmp_path, ["test.py"])
        assert patterns.get("docstring_style") == "sphinx"

    def test_detect_future_annotations(self, tmp_path: Path) -> None:
        """Detect future annotations import."""
        content = dedent("""
            from __future__ import annotations

            def foo() -> str:
                return "bar"
        """)
        (tmp_path / "test.py").write_text(content)

        patterns = _extract_code_patterns(tmp_path, ["test.py"])
        assert patterns.get("future_annotations") == "yes"

    def test_detect_type_hints(self, tmp_path: Path) -> None:
        """Detect type hints usage."""
        content = dedent("""
            def foo(x: int) -> str:
                return str(x)
        """)
        (tmp_path / "test.py").write_text(content)

        patterns = _extract_code_patterns(tmp_path, ["test.py"])
        assert patterns.get("type_hints") == "yes"

    def test_detect_structured_logging(self, tmp_path: Path) -> None:
        """Detect structured logging pattern."""
        content = dedent("""
            from ai_infra.logging import get_logger

            logger = get_logger(__name__)
        """)
        (tmp_path / "test.py").write_text(content)

        patterns = _extract_code_patterns(tmp_path, ["test.py"])
        assert patterns.get("logging") == "structured"

    def test_skip_non_python_files(self, tmp_path: Path) -> None:
        """Skip non-Python files."""
        (tmp_path / "test.js").write_text("console.log('hello')")

        patterns = _extract_code_patterns(tmp_path, ["test.js"])
        assert patterns == {}

    def test_limit_to_first_three_files(self, tmp_path: Path) -> None:
        """Only check first 3 files."""
        for i in range(5):
            (tmp_path / f"test{i}.py").write_text("# no patterns")

        # Should not error with many files
        patterns = _extract_code_patterns(tmp_path, [f"test{i}.py" for i in range(5)])
        assert isinstance(patterns, dict)


class TestBuildTaskSummaries:
    """Tests for task summary building."""

    def test_build_from_run_memory(self) -> None:
        """Build summaries from run memory dict."""
        run_memory = {
            "1.1": {
                "summary": "Created main module",
                "files_created": ["src/main.py"],
                "status": "completed",
            },
            "1.2": {
                "summary": "Added tests",
                "files_created": ["tests/test_main.py"],
                "status": "completed",
            },
        }

        summaries = _build_task_summaries(run_memory)
        assert len(summaries) == 2
        assert "Task 1.1" in summaries[0]
        assert "main" in summaries[0].lower()
        assert "Task 1.2" in summaries[1]

    def test_truncate_long_summaries(self) -> None:
        """Truncate long summary text."""
        run_memory = {
            "1.1": {
                "summary": "A" * 200,  # Very long summary
                "files_created": [],
            },
        }

        summaries = _build_task_summaries(run_memory)
        assert len(summaries) == 1
        assert "..." in summaries[0]
        assert len(summaries[0]) < 200

    def test_include_files_in_summary(self) -> None:
        """Include file names in summary."""
        run_memory = {
            "1.1": {
                "summary": "Created files",
                "files_created": ["a.py", "b.py", "c.py", "d.py"],
            },
        }

        summaries = _build_task_summaries(run_memory)
        assert len(summaries) == 1
        assert "a.py" in summaries[0]
        assert "+1 more" in summaries[0]  # Only first 3 files shown

    def test_empty_run_memory(self) -> None:
        """Handle empty run memory."""
        summaries = _build_task_summaries({})
        assert summaries == []

    def test_skip_non_dict_entries(self) -> None:
        """Skip non-dict run memory entries."""
        run_memory = {
            "1.1": "not a dict",
            "1.2": {"summary": "Valid entry"},
        }

        summaries = _build_task_summaries(run_memory)
        assert len(summaries) == 1


class TestBuildSessionBrief:
    """Tests for session brief building."""

    def test_basic_session_brief(self) -> None:
        """Build basic session brief."""
        brief = _build_session_brief(
            tasks_completed=5,
            files_created=["a.py", "b.py"],
            project_type="python",
        )

        assert "Tasks completed: 5" in brief
        assert "a.py" in brief
        assert "python" in brief

    def test_truncate_many_files(self) -> None:
        """Truncate when many files created."""
        files = [f"file{i}.py" for i in range(10)]
        brief = _build_session_brief(
            tasks_completed=10,
            files_created=files,
            project_type="python",
        )

        assert "+5 more" in brief

    def test_no_files_yet(self) -> None:
        """Handle case with no files created."""
        brief = _build_session_brief(
            tasks_completed=0,
            files_created=[],
            project_type="unknown",
        )

        assert "none yet" in brief


class TestGetRelevantFiles:
    """Tests for relevant files selection."""

    def test_include_modified_files(self, tmp_path: Path) -> None:
        """Include recently modified files."""
        relevant = _get_relevant_files(
            tmp_path,
            files_modified=["a.py", "b.py"],
            max_files=10,
        )

        assert "a.py" in relevant
        assert "b.py" in relevant

    def test_include_key_project_files(self, tmp_path: Path) -> None:
        """Include key project files if they exist."""
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "README.md").write_text("")

        relevant = _get_relevant_files(tmp_path, [], max_files=10)

        assert "pyproject.toml" in relevant
        assert "README.md" in relevant

    def test_respect_max_files(self, tmp_path: Path) -> None:
        """Respect max_files limit."""
        files = [f"file{i}.py" for i in range(20)]

        relevant = _get_relevant_files(tmp_path, files, max_files=5)

        assert len(relevant) <= 5


class TestBuildFileContentsCache:
    """Tests for file contents cache building."""

    def test_cache_recent_files(self, tmp_path: Path) -> None:
        """Cache contents of recent files."""
        (tmp_path / "a.py").write_text("# file a")
        (tmp_path / "b.py").write_text("# file b")

        cache = _build_file_contents_cache(
            tmp_path,
            files=["a.py", "b.py"],
            max_files=5,
            max_lines=50,
        )

        assert "a.py" in cache
        assert "b.py" in cache
        assert "# file a" in cache["a.py"]

    def test_respect_max_files(self, tmp_path: Path) -> None:
        """Only cache up to max_files."""
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"# file {i}")

        cache = _build_file_contents_cache(
            tmp_path,
            files=[f"file{i}.py" for i in range(10)],
            max_files=3,
        )

        assert len(cache) <= 3

    def test_skip_nonexistent_files(self, tmp_path: Path) -> None:
        """Skip files that don't exist."""
        cache = _build_file_contents_cache(
            tmp_path,
            files=["nonexistent.py"],
            max_files=5,
        )

        assert cache == {}


class TestExecutionContextFormatForPrompt:
    """Tests for ExecutionContext.format_for_prompt method."""

    def test_format_with_patterns(self) -> None:
        """Format context with code patterns."""
        ctx = ExecutionContext(
            workspace=Path("/tmp"),
            established_patterns={"docstring_style": "google", "type_hints": "yes"},
        )

        formatted = ctx.format_for_prompt()

        assert "Code Patterns to Follow" in formatted
        assert "docstring_style: google" in formatted
        assert "type_hints: yes" in formatted

    def test_format_with_files_modified(self) -> None:
        """Format context with modified files list."""
        ctx = ExecutionContext(
            workspace=Path("/tmp"),
            files_modified=["src/main.py", "src/utils.py"],
        )

        formatted = ctx.format_for_prompt()

        assert "Files Created This Session" in formatted
        assert "src/main.py" in formatted

    def test_format_with_task_summaries(self) -> None:
        """Format context with previous task summaries."""
        ctx = ExecutionContext(
            workspace=Path("/tmp"),
            previous_task_summaries=[
                "Task 1.1: Created main module",
                "Task 1.2: Added tests",
            ],
        )

        formatted = ctx.format_for_prompt()

        assert "Task Summaries" in formatted
        assert "Task 1.1" in formatted

    def test_format_limits_summaries_to_five(self) -> None:
        """Only include last 5 task summaries."""
        ctx = ExecutionContext(
            workspace=Path("/tmp"),
            previous_task_summaries=[f"Task {i}" for i in range(10)],
        )

        formatted = ctx.format_for_prompt()

        # Should only have last 5
        assert "Task 5" in formatted
        assert "Task 9" in formatted
        # First ones should be omitted
        assert "Task 0" not in formatted

    def test_format_with_session_summary(self) -> None:
        """Format context with session summary."""
        ctx = ExecutionContext(
            workspace=Path("/tmp"),
            summary="Session Progress:\n- Tasks: 5\n- Files: 10",
        )

        formatted = ctx.format_for_prompt()

        assert "Session Context" in formatted
        assert "Tasks: 5" in formatted

    def test_format_empty_context(self) -> None:
        """Format empty context returns empty string."""
        ctx = ExecutionContext(workspace=Path("/tmp"))

        formatted = ctx.format_for_prompt()

        assert formatted == ""

    def test_format_combined_sections(self) -> None:
        """Format with multiple sections."""
        ctx = ExecutionContext(
            workspace=Path("/tmp"),
            files_modified=["src/main.py"],
            established_patterns={"type_hints": "yes"},
            previous_task_summaries=["Task 1.1: Done"],
            summary="Session in progress",
        )

        formatted = ctx.format_for_prompt()

        # All sections should be present
        assert "Code Patterns to Follow" in formatted
        assert "Files Created This Session" in formatted
        assert "Task Summaries" in formatted
        assert "Session Context" in formatted


class TestBuildExecutionContext:
    """Tests for the full _build_execution_context function."""

    def test_build_context_from_state(self, tmp_path: Path) -> None:
        """Build complete context from executor state."""
        # Create project file
        (tmp_path / "pyproject.toml").write_text("[project]")

        # Create a Python file with patterns
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text(
            dedent('''
            from __future__ import annotations

            def foo(x: int) -> str:
                """Convert to string.

                Args:
                    x: Input value.

                Returns:
                    String representation.
                """
                return str(x)
        ''')
        )

        state = {
            "roadmap_path": str(tmp_path / "ROADMAP.md"),
            "files_modified": ["src/main.py"],
            "run_memory": {
                "1.1": {"summary": "Created main", "files_created": ["src/main.py"]},
            },
            "dependencies": ["pytest", "httpx"],
        }

        context = _build_execution_context(state)

        # Verify all fields populated
        assert context.workspace == tmp_path
        assert context.project_type == "python"
        assert "src/main.py" in context.files_modified
        assert len(context.previous_task_summaries) == 1
        assert "google" in context.established_patterns.get("docstring_style", "")
        assert context.file_contents_cache.get("src/main.py") is not None
        assert "Tasks completed: 1" in context.summary

    def test_build_context_empty_state(self) -> None:
        """Build context from minimal state."""
        state = {}

        context = _build_execution_context(state)

        assert context.workspace == Path.cwd()
        assert context.files_modified == []
        assert context.previous_task_summaries == []
        assert context.established_patterns == {}


class TestExecutionContextToDict:
    """Tests for ExecutionContext serialization."""

    def test_to_dict_includes_new_fields(self) -> None:
        """to_dict includes Phase 16.5.5 fields."""
        ctx = ExecutionContext(
            workspace=Path("/tmp"),
            previous_task_summaries=["Task 1"],
            established_patterns={"style": "google"},
            file_contents_cache={"a.py": "content"},
        )

        data = ctx.to_dict()

        assert "previous_task_summaries" in data
        assert data["previous_task_summaries"] == ["Task 1"]
        assert "established_patterns" in data
        assert data["established_patterns"] == {"style": "google"}
        # file_contents_cache should only include keys (for size)
        assert "file_contents_cache" in data
        assert data["file_contents_cache"] == ["a.py"]
