"""Tests for write files node.

Phase 1.3: Tests for write.py - the separated file writing module.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_infra.executor.nodes.write import route_after_write, write_files_node

# =============================================================================
# Write Files Node Tests
# =============================================================================


class TestWriteFilesNode:
    """Tests for write_files_node function."""

    @pytest.mark.asyncio
    async def test_writes_single_file(self, tmp_path: Path):
        """Should write a single file to disk."""
        state = {
            "generated_code": {"app.py": "print('hello')"},
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert result["files_modified"] == ["app.py"]
        assert (tmp_path / "app.py").read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_writes_multiple_files(self, tmp_path: Path):
        """Should write multiple files to disk."""
        state = {
            "generated_code": {
                "src/app.py": "print('app')",
                "src/utils.py": "print('utils')",
                "tests/test_app.py": "print('test')",
            },
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert len(result["files_modified"]) == 3
        assert (tmp_path / "src" / "app.py").read_text() == "print('app')"
        assert (tmp_path / "src" / "utils.py").read_text() == "print('utils')"
        assert (tmp_path / "tests" / "test_app.py").read_text() == "print('test')"

    @pytest.mark.asyncio
    async def test_creates_directories(self, tmp_path: Path):
        """Should create nested directories as needed."""
        state = {
            "generated_code": {
                "deep/nested/dir/file.py": "content",
            },
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert (tmp_path / "deep" / "nested" / "dir" / "file.py").exists()

    @pytest.mark.asyncio
    async def test_handles_empty_generated_code(self):
        """Should handle empty generated_code gracefully."""
        state = {
            "generated_code": {},
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert result["files_modified"] == []

    @pytest.mark.asyncio
    async def test_handles_missing_generated_code(self):
        """Should handle missing generated_code field."""
        state = {"validated": True}

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert result["files_modified"] == []

    @pytest.mark.asyncio
    async def test_fails_if_not_validated(self, tmp_path: Path):
        """Should fail if code was not validated first."""
        state = {
            "generated_code": {"app.py": "print('hello')"},
            "project_root": str(tmp_path),
            "validated": False,
        }

        result = await write_files_node(state)

        assert result["files_written"] is False
        assert result["error"] is not None
        assert result["error"]["error_type"] == "validation"
        assert "not validated" in result["error"]["message"]
        # File should NOT be written
        assert not (tmp_path / "app.py").exists()

    @pytest.mark.asyncio
    async def test_fails_if_validated_missing(self, tmp_path: Path):
        """Should fail if validated field is missing."""
        state = {
            "generated_code": {"app.py": "print('hello')"},
            "project_root": str(tmp_path),
        }

        result = await write_files_node(state)

        assert result["files_written"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self, tmp_path: Path):
        """Dry run should not write files to disk."""
        state = {
            "generated_code": {"app.py": "print('hello')"},
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state, dry_run=True)

        assert result["files_written"] is True
        assert result["files_modified"] == ["app.py"]
        # File should NOT be written in dry run
        assert not (tmp_path / "app.py").exists()

    @pytest.mark.asyncio
    async def test_uses_roadmap_path_as_fallback(self, tmp_path: Path):
        """Should use roadmap_path parent as project_root if not specified."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# Roadmap")

        state = {
            "generated_code": {"app.py": "print('hello')"},
            "roadmap_path": str(roadmap),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert (tmp_path / "app.py").exists()

    @pytest.mark.asyncio
    async def test_handles_absolute_paths(self, tmp_path: Path):
        """Should handle absolute file paths."""
        abs_path = str(tmp_path / "absolute" / "path" / "file.py")
        state = {
            "generated_code": {abs_path: "content"},
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert Path(abs_path).exists()
        assert Path(abs_path).read_text() == "content"

    @pytest.mark.asyncio
    async def test_tracks_task_id_in_logs(self, tmp_path: Path):
        """Should use task ID from current_task for logging."""
        task = MagicMock()
        task.id = "1.2.3"

        state = {
            "generated_code": {"app.py": "print('hello')"},
            "project_root": str(tmp_path),
            "validated": True,
            "current_task": task,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True

    @pytest.mark.asyncio
    async def test_overwrites_existing_files(self, tmp_path: Path):
        """Should overwrite existing files."""
        existing = tmp_path / "app.py"
        existing.write_text("old content")

        state = {
            "generated_code": {"app.py": "new content"},
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        assert existing.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_handles_write_errors(self, tmp_path: Path):
        """Should handle write errors gracefully."""
        # Create a directory where we'll try to write a file
        # (can't write file with same name as existing dir)
        (tmp_path / "app.py").mkdir()

        state = {
            "generated_code": {"app.py": "content"},
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is False
        assert result["error"] is not None
        assert result["write_errors"] is not None
        assert len(result["write_errors"]) == 1

    @pytest.mark.asyncio
    async def test_partial_write_failure(self, tmp_path: Path):
        """Should continue writing other files if one fails."""
        # Create a directory where we'll try to write a file
        (tmp_path / "fail.py").mkdir()

        state = {
            "generated_code": {
                "success.py": "good content",
                "fail.py": "bad content",
            },
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        # One file succeeded
        assert "success.py" in result["files_modified"]
        assert (tmp_path / "success.py").read_text() == "good content"
        # But we have errors
        assert result["error"] is not None
        assert len(result["write_errors"]) == 1

    @pytest.mark.asyncio
    async def test_writes_utf8_content(self, tmp_path: Path):
        """Should write UTF-8 encoded content."""
        state = {
            "generated_code": {"app.py": "# -*- coding: utf-8 -*-\nprint('Hello World')"},
            "project_root": str(tmp_path),
            "validated": True,
        }

        result = await write_files_node(state)

        assert result["files_written"] is True
        content = (tmp_path / "app.py").read_text(encoding="utf-8")
        assert "Hello World" in content


# =============================================================================
# Route Function Tests
# =============================================================================


class TestRouteAfterWrite:
    """Tests for route_after_write function."""

    def test_success_routes_to_verify(self):
        """When files_written is True, should route to verify_task."""
        state = {"files_written": True, "error": None}
        assert route_after_write(state) == "verify_task"

    def test_no_error_routes_to_verify(self):
        """When no error, should route to verify_task."""
        state = {}
        assert route_after_write(state) == "verify_task"

    def test_error_routes_to_failure(self):
        """When error is present, should route to handle_failure."""
        state = {"error": {"message": "write failed"}}
        assert route_after_write(state) == "handle_failure"

    def test_with_write_errors_but_no_error_goes_to_verify(self):
        """Write errors without main error should still verify."""
        state = {
            "files_written": True,
            "write_errors": [{"file": "x.py", "error": "minor"}],
            "error": None,
        }
        assert route_after_write(state) == "verify_task"
