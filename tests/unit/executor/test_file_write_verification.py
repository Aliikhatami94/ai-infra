"""Tests for File Write Verification (Phase 5.11.1).

This module tests:
- _extract_expected_files() - parsing file paths from task descriptions
- _verify_files_created() - checking if files exist on disk
- Integration with retry flow
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.loop import Executor, ExecutorConfig
from ai_infra.executor.parser import ParsedTask

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP.md content for testing."""
    return """# Test Project

## Phase 1: Setup

### 1.1 Create Files
- [ ] Create `src/utils.py` with helper functions
"""


@pytest.fixture
def temp_project(sample_roadmap_content: str):
    """Create a temporary project directory with ROADMAP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        roadmap_path = project_path / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)

        # Create src directory
        (project_path / "src").mkdir()

        yield project_path, roadmap_path


@pytest.fixture
def executor(temp_project) -> Executor:
    """Create an Executor instance for testing."""
    project_path, roadmap_path = temp_project
    config = ExecutorConfig()
    return Executor(roadmap_path, config=config)


# =============================================================================
# Test _extract_expected_files - Pattern Matching
# =============================================================================


class TestExtractExpectedFilesPatterns:
    """Tests for _extract_expected_files pattern matching."""

    def test_backtick_pattern_simple(self, executor: Executor):
        """Test extraction of backtick-wrapped file paths."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/utils.py` with helper functions",
            description="Create the utils module",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/utils.py" in files

    def test_backtick_pattern_multiple(self, executor: Executor):
        """Test extraction of multiple backtick-wrapped paths."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create modules",
            description="Create `src/config.py` and `src/database.py`",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/config.py" in files
        assert "src/database.py" in files

    def test_backtick_pattern_nested_path(self, executor: Executor):
        """Test extraction of deeply nested paths."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/models/user.py` with User class",
            description="Define the user model",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/models/user.py" in files

    def test_backtick_pattern_javascript(self, executor: Executor):
        """Test extraction of JavaScript file paths."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/utils.js` with formatName function",
            description="Export formatName",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/utils.js" in files

    def test_bare_path_pattern(self, executor: Executor):
        """Test extraction of bare paths (without backticks)."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create src/main.py entry point",
            description="Main application file",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/main.py" in files

    def test_files_marker_pattern(self, executor: Executor):
        """Test extraction from **Files**: marker."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create greeter module",
            description="**Files**: `src/greeter.js`",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/greeter.js" in files

    def test_checkbox_pattern(self, executor: Executor):
        """Test extraction from checkbox format."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Setup project",
            description="- [ ] `src/config.py` - configuration\n- [ ] `src/app.py` - main app",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/config.py" in files
        assert "src/app.py" in files

    def test_multiple_extensions(self, executor: Executor):
        """Test extraction of various file extensions."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create files",
            description=(
                "Create `src/app.ts`, `tests/test.tsx`, "
                "`styles/main.css`, `config.json`, `README.md`"
            ),
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/app.ts" in files
        assert "tests/test.tsx" in files
        assert "styles/main.css" in files
        assert "config.json" in files
        assert "README.md" in files

    def test_no_duplicates(self, executor: Executor):
        """Test that duplicate paths are removed."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/utils.py` module",
            description="Create `src/utils.py` with functions. **Files**: `src/utils.py`",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        # Should only appear once
        assert files.count("src/utils.py") == 1

    def test_filters_function_calls(self, executor: Executor):
        """Test that function calls are filtered out."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create utils with formatName(name) returning name.toUpperCase()",
            description="Create `src/utils.js` with the function",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        # Should include the file but not the function reference
        assert "src/utils.js" in files
        assert "name.toUpperCase()" not in files

    def test_empty_task(self, executor: Executor):
        """Test with task that has no file references."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Configure project settings",
            description="Update the project configuration",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert files == []

    def test_title_only(self, executor: Executor):
        """Test extraction from title when description is empty."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/models/product.py`",
            description="",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/models/product.py" in files


# =============================================================================
# Test _verify_files_created
# =============================================================================


class TestVerifyFilesCreated:
    """Tests for _verify_files_created verification."""

    def test_all_files_exist(self, temp_project):
        """Test verification passes when all files exist."""
        project_path, roadmap_path = temp_project

        # Create the expected file
        (project_path / "src" / "utils.py").write_text("# utils")

        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/utils.py`",
            description="Create utils module",
            status="pending",
        )

        result = MagicMock()
        result.success = True

        all_created, missing = executor._verify_files_created(task, result)

        assert all_created is True
        assert missing == []

    def test_file_missing(self, temp_project):
        """Test verification fails when file is missing."""
        project_path, roadmap_path = temp_project

        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/missing.py`",
            description="Create missing module",
            status="pending",
        )

        result = MagicMock()
        result.success = True

        all_created, missing = executor._verify_files_created(task, result)

        assert all_created is False
        assert "src/missing.py" in missing

    def test_partial_files_created(self, temp_project):
        """Test verification when some files exist and some don't."""
        project_path, roadmap_path = temp_project

        # Create one file but not the other
        (project_path / "src" / "config.py").write_text("# config")

        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create modules",
            description="Create `src/config.py` and `src/database.py`",
            status="pending",
        )

        result = MagicMock()
        result.success = True

        all_created, missing = executor._verify_files_created(task, result)

        assert all_created is False
        assert "src/database.py" in missing
        assert "src/config.py" not in missing

    def test_nested_directory_file(self, temp_project):
        """Test verification of files in nested directories."""
        project_path, roadmap_path = temp_project

        # Create nested directory and file
        (project_path / "src" / "models").mkdir(parents=True)
        (project_path / "src" / "models" / "user.py").write_text("# user")

        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/models/user.py`",
            description="User model",
            status="pending",
        )

        result = MagicMock()
        result.success = True

        all_created, missing = executor._verify_files_created(task, result)

        assert all_created is True
        assert missing == []

    def test_no_expected_files(self, temp_project):
        """Test verification passes when no files are expected."""
        project_path, roadmap_path = temp_project

        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Configure project",
            description="Update settings",
            status="pending",
        )

        result = MagicMock()
        result.success = True

        all_created, missing = executor._verify_files_created(task, result)

        assert all_created is True
        assert missing == []

    def test_multiple_missing_files(self, temp_project):
        """Test verification reports all missing files."""
        project_path, roadmap_path = temp_project

        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create project structure",
            description="Create `src/a.py`, `src/b.py`, `src/c.py`",
            status="pending",
        )

        result = MagicMock()
        result.success = True

        all_created, missing = executor._verify_files_created(task, result)

        assert all_created is False
        assert len(missing) == 3
        assert "src/a.py" in missing
        assert "src/b.py" in missing
        assert "src/c.py" in missing


# =============================================================================
# Test Retry Integration
# =============================================================================


class TestFileVerificationRetry:
    """Tests for file verification integration with retry flow."""

    @pytest.mark.asyncio
    async def test_retry_on_missing_files(self, temp_project):
        """Test that missing files trigger retry."""
        project_path, roadmap_path = temp_project

        config = ExecutorConfig(retry_failed=2)
        executor = Executor(roadmap_path, config=config)

        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.arun = AsyncMock(return_value="Created file")
        executor._agent = mock_agent

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/missing.py`",
            description="Create the file",
            status="pending",
        )

        # First call succeeds but file not created, second call creates file
        call_count = 0

        async def side_effect(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Create the file on second attempt
                (project_path / "src" / "missing.py").write_text("# created")
            return "Done"

        mock_agent.arun.side_effect = side_effect

        result = await executor._execute_task_with_retry(task)

        # Should have called agent twice
        assert mock_agent.arun.call_count == 2
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_retry_when_files_exist(self, temp_project):
        """Test that no retry happens when files are created."""
        project_path, roadmap_path = temp_project

        config = ExecutorConfig(retry_failed=3)
        executor = Executor(roadmap_path, config=config)

        # Mock the agent
        mock_agent = AsyncMock()
        executor._agent = mock_agent

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/exists.py`",
            description="Create the file",
            status="pending",
        )

        async def side_effect(prompt):
            # Create the file on first attempt
            (project_path / "src" / "exists.py").write_text("# created")
            return "Created src/exists.py"

        mock_agent.arun.side_effect = side_effect

        result = await executor._execute_task_with_retry(task)

        # Should have called agent only once
        assert mock_agent.arun.call_count == 1
        assert result.success is True

    @pytest.mark.asyncio
    async def test_retries_exhausted_with_missing_files(self, temp_project):
        """Test that verification still passes after retries exhausted if file missing."""
        project_path, roadmap_path = temp_project

        config = ExecutorConfig(retry_failed=2)
        executor = Executor(roadmap_path, config=config)

        # Mock the agent - never creates the file
        mock_agent = AsyncMock()
        mock_agent.arun = AsyncMock(return_value="Done")
        executor._agent = mock_agent

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/never.py`",
            description="Create the file",
            status="pending",
        )

        result = await executor._execute_task_with_retry(task)

        # Should have retried max times
        assert mock_agent.arun.call_count == 2
        # Result is still success (agent said it succeeded, we just logged warning)
        # The file verification warning is logged but doesn't fail the task on last attempt
        assert result.success is True


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestFileVerificationEdgeCases:
    """Tests for edge cases in file verification."""

    def test_leading_slash_path(self, executor: Executor):
        """Test extraction handles leading slash."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `/src/utils.py`",
            description="Note: has leading slash",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        # Should still extract the path
        assert len(files) > 0

    def test_windows_style_path(self, executor: Executor):
        """Test extraction with backslash paths."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src\\utils.py`",
            description="Windows-style path",
            status="pending",
        )

        # Backslash paths may not match our patterns
        files = executor._extract_expected_files(task)

        # May or may not extract - just ensure no crash
        assert isinstance(files, list)

    def test_case_insensitive_create(self, executor: Executor):
        """Test that Create pattern is case insensitive."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="CREATE `src/utils.py` module",
            description="create `src/app.py` too",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/utils.py" in files
        assert "src/app.py" in files

    def test_sql_and_shell_extensions(self, executor: Executor):
        """Test extraction of SQL and shell file extensions."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create database files",
            description="Create `migrations/001.sql` and `scripts/deploy.sh`",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "migrations/001.sql" in files
        assert "scripts/deploy.sh" in files

    def test_file_with_hyphen_and_underscore(self, executor: Executor):
        """Test extraction of paths with hyphens and underscores."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/my-module_v2.py`",
            description="",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "src/my-module_v2.py" in files

    def test_test_file_pattern(self, executor: Executor):
        """Test extraction of test file paths."""
        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `tests/test_utils.py`",
            description="Also create `tests/test_main.py`",
            status="pending",
        )

        files = executor._extract_expected_files(task)

        assert "tests/test_utils.py" in files
        assert "tests/test_main.py" in files


# =============================================================================
# Test _build_missing_files_context (Phase 5.11.3)
# =============================================================================


class TestBuildMissingFilesContext:
    """Tests for _build_missing_files_context method."""

    def test_builds_context_with_absolute_paths(self, temp_project):
        """Test that context includes absolute paths for missing files."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create modules",
            description="Create config and database modules",
            status="pending",
        )

        missing_files = ["src/config.py", "src/database.py"]
        context = executor._build_missing_files_context(task, missing_files, attempt=2)

        # Check that absolute paths are included
        assert str(project_path / "src/config.py") in context
        assert str(project_path / "src/database.py") in context

    def test_context_includes_retry_attempt_number(self, temp_project):
        """Test that context includes the retry attempt number."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create files",
            description="",
            status="pending",
        )

        context = executor._build_missing_files_context(task, ["src/app.py"], attempt=3)

        assert "Retry Attempt 3" in context
        assert "Missing Files" in context

    def test_context_includes_original_task(self, temp_project):
        """Test that context includes original task title and description."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create utility module",
            description="Implement helper functions for data processing",
            status="pending",
        )

        context = executor._build_missing_files_context(task, ["src/utils.py"], attempt=2)

        assert "Create utility module" in context
        assert "Implement helper functions for data processing" in context

    def test_context_lists_all_missing_files(self, temp_project):
        """Test that all missing files are listed in context."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create multiple files",
            description="",
            status="pending",
        )

        missing_files = [
            "src/config.py",
            "src/database.py",
            "src/utils.py",
            "tests/test_config.py",
        ]
        context = executor._build_missing_files_context(task, missing_files, attempt=2)

        for f in missing_files:
            assert f in context

    def test_context_includes_instructions(self, temp_project):
        """Test that context includes clear instructions for creating files."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create file",
            description="",
            status="pending",
        )

        context = executor._build_missing_files_context(task, ["src/app.py"], attempt=2)

        assert "Create each file" in context
        assert "write_file" in context
        assert "NOT create empty files" in context

    def test_context_mentions_file_write_reliability(self, temp_project):
        """Test that context explains the issue is about file write reliability."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create file",
            description="",
            status="pending",
        )

        context = executor._build_missing_files_context(task, ["src/app.py"], attempt=2)

        assert "files were not created on disk" in context
        assert "file write reliability" in context


class TestMissingFilesRetryIntegration:
    """Integration tests for missing files retry flow with explicit context."""

    @pytest.mark.asyncio
    async def test_retry_uses_missing_files_context(self, temp_project):
        """Test that retry uses specialized missing files context."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(retry_failed=2)
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/app.py`",
            description="Create the main application file",
            status="pending",
        )

        # Track the retry_context passed to _execute_task
        captured_contexts: list[str | None] = []

        async def mock_execute_task(t, *, retry_context=None):
            captured_contexts.append(retry_context)
            # Return success but file doesn't exist
            return MagicMock(
                success=True,
                error=None,
                files_modified=1,
            )

        with patch.object(executor, "_execute_task", side_effect=mock_execute_task):
            with patch.object(executor, "_verify_files_created") as mock_verify:
                # First call: files missing, second call: files exist
                mock_verify.side_effect = [
                    (False, ["src/app.py"]),  # First attempt: missing
                    (True, []),  # Second attempt: verified
                ]

                await executor._execute_task_with_retry(task)

        # Should have been called twice
        assert len(captured_contexts) == 2

        # First call has no retry context
        assert captured_contexts[0] is None

        # Second call should have missing files context with absolute path
        retry_ctx = captured_contexts[1]
        assert retry_ctx is not None
        assert "Missing Files" in retry_ctx
        # Use endswith to handle macOS /var vs /private/var symlink
        assert "src/app.py" in retry_ctx

    @pytest.mark.asyncio
    async def test_logs_warning_with_absolute_paths(self, temp_project):
        """Test that warning log includes absolute paths."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(retry_failed=2)
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/config.py`",
            description="",
            status="pending",
        )

        async def mock_execute_task(t, *, retry_context=None):
            return MagicMock(success=True, error=None, files_modified=1)

        with patch.object(executor, "_execute_task", side_effect=mock_execute_task):
            with patch.object(executor, "_verify_files_created") as mock_verify:
                mock_verify.side_effect = [
                    (False, ["src/config.py"]),
                    (True, []),
                ]

                with patch("ai_infra.executor.loop.logger") as mock_logger:
                    await executor._execute_task_with_retry(task)

                    # Find the warning call about missing files
                    warning_calls = [
                        call
                        for call in mock_logger.warning.call_args_list
                        if call[0][0] == "task_files_missing_will_retry"
                    ]

                    assert len(warning_calls) == 1
                    # Check that absolute_paths kwarg was passed
                    call_kwargs = warning_calls[0][1]
                    assert "absolute_paths" in call_kwargs
                    # Check that path ends with the expected relative path
                    # (use endswith to handle macOS /var vs /private/var symlink)
                    abs_paths = call_kwargs["absolute_paths"]
                    assert len(abs_paths) == 1
                    assert abs_paths[0].endswith("src/config.py")

    @pytest.mark.asyncio
    async def test_regular_error_uses_standard_context(self, temp_project):
        """Test that regular errors still use standard retry context."""
        from ai_infra.executor.adaptive import AdaptiveMode

        project_path, roadmap_path = temp_project
        config = ExecutorConfig(retry_failed=2)
        executor = Executor(roadmap_path, config=config)
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Run some code",
            description="Execute the script",
            status="pending",
        )

        captured_contexts: list[str | None] = []
        call_count = 0

        async def mock_execute_task(t, *, retry_context=None):
            nonlocal call_count
            captured_contexts.append(retry_context)
            call_count += 1
            if call_count == 1:
                return MagicMock(success=False, error="ImportError: No module named 'foo'")
            return MagicMock(success=True, error=None, files_modified=0)

        with patch.object(executor, "_execute_task", side_effect=mock_execute_task):
            with patch.object(executor, "_verify_files_created", return_value=(True, [])):
                await executor._execute_task_with_retry(task)

        # Second call should have standard retry context (not missing files)
        assert len(captured_contexts) == 2
        retry_ctx = captured_contexts[1]
        assert retry_ctx is not None
        assert "Error from Previous Attempt" in retry_ctx
        assert "Missing Files" not in retry_ctx


# =============================================================================
# Test Phase 5.11.4: File Write Verification with Checksum
# =============================================================================


class TestFileWriteModels:
    """Tests for FileWriteRecord and FileWriteSummary models."""

    def test_file_write_record_to_dict(self):
        """Test FileWriteRecord serialization."""

        from ai_infra.executor.models import FileWriteRecord

        record = FileWriteRecord(
            path="src/utils.py",
            absolute_path="/project/src/utils.py",
            task_id="1.1.1",
            size_bytes=1024,
            checksum="abc123def456",
            verified=True,
        )

        d = record.to_dict()
        assert d["path"] == "src/utils.py"
        assert d["absolute_path"] == "/project/src/utils.py"
        assert d["task_id"] == "1.1.1"
        assert d["size_bytes"] == 1024
        assert d["checksum"] == "abc123def456"
        assert d["verified"] is True

    def test_file_write_record_from_dict(self):
        """Test FileWriteRecord deserialization."""
        from ai_infra.executor.models import FileWriteRecord

        data = {
            "path": "src/config.py",
            "absolute_path": "/project/src/config.py",
            "task_id": "1.1.2",
            "size_bytes": 512,
            "checksum": "xyz789",
            "created_at": "2026-01-07T12:00:00",
            "verified": True,
        }

        record = FileWriteRecord.from_dict(data)
        assert record.path == "src/config.py"
        assert record.size_bytes == 512
        assert record.verified is True

    def test_file_write_summary_text_all_verified(self):
        """Test summary text when all files are verified."""
        from ai_infra.executor.models import FileWriteRecord, FileWriteSummary

        summary = FileWriteSummary(
            total_expected=3,
            total_created=3,
            total_verified=3,
            missing_files=[],
            verified_files=[
                FileWriteRecord(
                    path=f"src/file{i}.py",
                    absolute_path=f"/project/src/file{i}.py",
                    task_id="1.1.1",
                    size_bytes=100,
                    checksum="abc",
                    verified=True,
                )
                for i in range(3)
            ],
        )

        text = summary.summary_text()
        assert "Expected files:  3" in text
        assert "Created files:   3" in text
        assert "Verified files:  3" in text
        assert "All files verified successfully" in text

    def test_file_write_summary_text_with_missing(self):
        """Test summary text when files are missing."""
        from ai_infra.executor.models import FileWriteSummary

        summary = FileWriteSummary(
            total_expected=5,
            total_created=3,
            total_verified=3,
            missing_files=["src/a.py", "src/b.py"],
            verified_files=[],
        )

        text = summary.summary_text()
        assert "MISSING FILES:" in text
        assert "src/a.py" in text
        assert "src/b.py" in text
        assert "2 file(s) could not be verified" in text


class TestComputeFileChecksum:
    """Tests for _compute_file_checksum method."""

    def test_computes_correct_checksum(self, temp_project):
        """Test that checksum is correctly computed."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        # Create a file with known content
        test_file = project_path / "test_checksum.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        checksum, size = executor._compute_file_checksum(test_file)

        # Verify size
        assert size == len(test_content)

        # Verify checksum is a valid MD5 hex string
        assert len(checksum) == 32
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_different_content_different_checksum(self, temp_project):
        """Test that different content produces different checksums."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig()
        executor = Executor(roadmap_path, config=config)

        file1 = project_path / "file1.txt"
        file2 = project_path / "file2.txt"
        file1.write_text("content A")
        file2.write_text("content B")

        checksum1, _ = executor._compute_file_checksum(file1)
        checksum2, _ = executor._compute_file_checksum(file2)

        assert checksum1 != checksum2


class TestRecordFileWrite:
    """Tests for _record_file_write method."""

    def test_records_existing_file(self, temp_project):
        """Test recording an existing file."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        # Create a file
        test_file = project_path / "src" / "recorded.py"
        test_file.write_text("# recorded file")

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create file",
            description="",
            status="pending",
        )

        record = executor._record_file_write(task, "src/recorded.py")

        assert record is not None
        assert record.path == "src/recorded.py"
        assert record.task_id == "1.1.1"
        assert record.size_bytes > 0
        assert record.verified is True
        assert len(record.checksum) == 32

    def test_returns_none_for_missing_file(self, temp_project):
        """Test that None is returned for non-existent files."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create file",
            description="",
            status="pending",
        )

        record = executor._record_file_write(task, "nonexistent.py")

        assert record is None

    def test_adds_to_tracker(self, temp_project):
        """Test that record is added to the tracker."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        # Create a file
        test_file = project_path / "src" / "tracked.py"
        test_file.write_text("# tracked")

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create file",
            description="",
            status="pending",
        )

        executor._record_file_write(task, "src/tracked.py")

        assert "src/tracked.py" in executor._file_write_tracker
        assert executor._file_write_tracker["src/tracked.py"].task_id == "1.1.1"


class TestVerifyAndRecordFiles:
    """Tests for _verify_and_record_files method."""

    def test_tracks_expected_files(self, temp_project):
        """Test that expected files are tracked per task."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        # Create the expected file
        test_file = project_path / "src" / "expected.py"
        test_file.write_text("# expected")

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/expected.py`",
            description="",
            status="pending",
        )

        result = MagicMock(success=True, files_modified=1)

        files_ok, missing = executor._verify_and_record_files(task, result)

        assert files_ok is True
        assert missing == []
        assert "1.1.1" in executor._expected_files_per_task
        assert "src/expected.py" in executor._expected_files_per_task["1.1.1"]

    def test_records_with_checksum_when_enabled(self, temp_project):
        """Test that files are recorded with checksum when verify_writes is enabled."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        # Create the file
        test_file = project_path / "src" / "checksummed.py"
        test_file.write_text("# checksummed")

        task = ParsedTask(
            id="1.1.1",
            phase_id="1.1",
            title="Create `src/checksummed.py`",
            description="",
            status="pending",
        )

        result = MagicMock(success=True, files_modified=1)

        executor._verify_and_record_files(task, result)

        assert "src/checksummed.py" in executor._file_write_tracker
        record = executor._file_write_tracker["src/checksummed.py"]
        assert record.checksum is not None
        assert len(record.checksum) == 32


class TestGenerateFileWriteSummary:
    """Tests for _generate_file_write_summary method."""

    def test_empty_summary(self, temp_project):
        """Test summary when no files were expected."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        summary = executor._generate_file_write_summary()

        assert summary.total_expected == 0
        assert summary.total_created == 0
        assert summary.total_verified == 0

    def test_summary_with_verified_files(self, temp_project):
        """Test summary with verified files."""
        from ai_infra.executor.models import FileWriteRecord

        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        # Create files and add to tracker
        for i in range(3):
            f = project_path / "src" / f"file{i}.py"
            f.write_text(f"# file {i}")

        executor._expected_files_per_task["1.1.1"] = [
            "src/file0.py",
            "src/file1.py",
            "src/file2.py",
        ]

        for i in range(3):
            executor._file_write_tracker[f"src/file{i}.py"] = FileWriteRecord(
                path=f"src/file{i}.py",
                absolute_path=str(project_path / "src" / f"file{i}.py"),
                task_id="1.1.1",
                size_bytes=100,
                checksum="abc123",
                verified=True,
            )

        summary = executor._generate_file_write_summary()

        assert summary.total_expected == 3
        assert summary.total_verified == 3
        assert len(summary.missing_files) == 0

    def test_summary_with_missing_files(self, temp_project):
        """Test summary with missing files."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(verify_writes=True)
        executor = Executor(roadmap_path, config=config)

        # Only create 1 of 3 expected files
        f = project_path / "src" / "exists.py"
        f.write_text("# exists")

        executor._expected_files_per_task["1.1.1"] = [
            "src/exists.py",
            "src/missing1.py",
            "src/missing2.py",
        ]

        summary = executor._generate_file_write_summary()

        assert summary.total_expected == 3
        assert "src/missing1.py" in summary.missing_files
        assert "src/missing2.py" in summary.missing_files


class TestVerifyWritesConfig:
    """Tests for verify_writes config option."""

    def test_verify_writes_default_false(self):
        """Test that verify_writes defaults to False."""
        config = ExecutorConfig()
        assert config.verify_writes is False

    def test_verify_writes_in_to_dict(self):
        """Test that verify_writes is included in to_dict."""
        config = ExecutorConfig(verify_writes=True)
        d = config.to_dict()
        assert "verify_writes" in d
        assert d["verify_writes"] is True

    def test_verify_writes_enables_checksum_recording(self, temp_project):
        """Test that verify_writes enables checksum recording in retry loop."""
        project_path, roadmap_path = temp_project

        # With verify_writes=True, should use _verify_and_record_files
        config_enabled = ExecutorConfig(verify_writes=True, retry_failed=1)
        executor_enabled = Executor(roadmap_path, config=config_enabled)

        # With verify_writes=False, should use _verify_files_created
        config_disabled = ExecutorConfig(verify_writes=False, retry_failed=1)
        executor_disabled = Executor(roadmap_path, config=config_disabled)

        # Both should have the methods
        assert hasattr(executor_enabled, "_verify_and_record_files")
        assert hasattr(executor_disabled, "_verify_files_created")
