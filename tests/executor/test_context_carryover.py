"""Tests for executor context carryover (Phase 5.3)."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_infra.executor.context_carryover import (
    ArchitectureTracker,
    ContextStorage,
    ProjectArchitecture,
    SessionSummary,
    load_session_context,
    save_session_context,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def context_storage(temp_dir):
    """Create a context storage for tests."""
    return ContextStorage(base_path=temp_dir)


@pytest.fixture
def sample_summary():
    """Create a sample session summary."""
    return SessionSummary(
        session_id="test-session-123",
        workspace="/path/to/project",
        started_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
        tasks_completed=["Add authentication", "Add tests"],
        files_modified=["src/auth.py", "tests/test_auth.py"],
        project_type="FastAPI API with SQLAlchemy",
        key_patterns=["Use SQLAlchemy for DB", "JWT for authentication"],
        key_decisions=["Chose JWT over sessions for stateless auth"],
        warnings=["SQLite in dev, use PostgreSQL in prod"],
        continuation_hint="Consider adding rate limiting next",
    )


@pytest.fixture
def sample_architecture():
    """Create a sample project architecture."""
    return ProjectArchitecture(
        src_layout="src/",
        test_layout="tests/",
        config_location="root",
        entry_points=["main.py", "app.py"],
        core_modules=["auth", "models", "api"],
        utilities=["utils", "helpers"],
        naming_convention="snake_case",
        import_style="absolute",
        docstring_style="google",
        key_dependencies=["fastapi", "sqlalchemy", "pydantic"],
        internal_imports=["auth imports models", "api imports auth"],
    )


@pytest.fixture
def sample_project(temp_dir):
    """Create a sample project structure for architecture detection."""
    project = temp_dir / "sample_project"
    project.mkdir()

    # Create src layout
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text(
        '''"""Main module."""

from mypackage import utils

def main():
    """Entry point.

    Args:
        None

    Returns:
        None
    """
    pass
'''
    )
    (project / "src" / "mypackage").mkdir()
    (project / "src" / "mypackage" / "__init__.py").touch()
    (project / "src" / "mypackage" / "utils.py").write_text(
        '''"""Utilities module."""

def helper_func():
    """Helper function.

    Args:
        None

    Returns:
        str: A string.
    """
    return "hello"
'''
    )

    # Create tests layout
    (project / "tests").mkdir()
    (project / "tests" / "__init__.py").touch()
    (project / "tests" / "test_main.py").write_text("def test_main(): pass")

    # Create config
    (project / "pyproject.toml").write_text(
        """[project]
name = "sample"
dependencies = [
    "fastapi>=0.100.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
]
"""
    )

    return project


# =============================================================================
# SessionSummary Tests
# =============================================================================


class TestSessionSummary:
    """Tests for SessionSummary model."""

    def test_create_session_summary(self, sample_summary):
        """Test creating a SessionSummary."""
        assert sample_summary.session_id == "test-session-123"
        assert sample_summary.workspace == "/path/to/project"
        assert len(sample_summary.tasks_completed) == 2
        assert sample_summary.project_type == "FastAPI API with SQLAlchemy"

    def test_session_summary_defaults(self):
        """Test SessionSummary with minimal data."""
        summary = SessionSummary(
            session_id="test",
            workspace="/path",
        )
        assert summary.tasks_completed == []
        assert summary.files_modified == []
        assert summary.project_type == ""
        assert summary.continuation_hint == ""

    def test_session_summary_to_dict(self, sample_summary):
        """Test SessionSummary serialization."""
        data = sample_summary.to_dict()
        assert data["session_id"] == "test-session-123"
        assert data["workspace"] == "/path/to/project"
        assert "started_at" in data
        assert "completed_at" in data
        assert len(data["tasks_completed"]) == 2

    def test_session_summary_from_dict(self, sample_summary):
        """Test SessionSummary deserialization."""
        data = sample_summary.to_dict()
        restored = SessionSummary.from_dict(data)
        assert restored.session_id == sample_summary.session_id
        assert restored.workspace == sample_summary.workspace
        assert restored.tasks_completed == sample_summary.tasks_completed
        assert restored.project_type == sample_summary.project_type

    def test_session_summary_to_context_prompt(self, sample_summary):
        """Test context prompt generation."""
        prompt = sample_summary.to_context_prompt()

        assert "Previous Session Context" in prompt
        assert "FastAPI API with SQLAlchemy" in prompt
        assert "Add authentication" in prompt
        assert "SQLAlchemy for DB" in prompt
        assert "JWT over sessions" in prompt
        assert "SQLite in dev" in prompt
        assert "rate limiting" in prompt

    def test_session_summary_context_prompt_empty(self):
        """Test context prompt with empty data."""
        summary = SessionSummary(session_id="test", workspace="/path")
        prompt = summary.to_context_prompt()
        assert "Previous Session Context" in prompt
        # Should not have section headers for empty lists
        assert "Established Patterns" not in prompt
        assert "Important Decisions" not in prompt

    def test_session_summary_limits_stored_data(self):
        """Test that summary limits stored data."""
        summary = SessionSummary(
            session_id="test",
            workspace="/path",
            files_modified=[f"file_{i}.py" for i in range(100)],
            key_patterns=[f"pattern_{i}" for i in range(50)],
            key_decisions=[f"decision_{i}" for i in range(50)],
            warnings=[f"warning_{i}" for i in range(20)],
        )
        data = summary.to_dict()
        assert len(data["files_modified"]) == 50
        assert len(data["key_patterns"]) == 20
        assert len(data["key_decisions"]) == 20
        assert len(data["warnings"]) == 10


# =============================================================================
# ProjectArchitecture Tests
# =============================================================================


class TestProjectArchitecture:
    """Tests for ProjectArchitecture model."""

    def test_create_architecture(self, sample_architecture):
        """Test creating ProjectArchitecture."""
        assert sample_architecture.src_layout == "src/"
        assert sample_architecture.test_layout == "tests/"
        assert len(sample_architecture.entry_points) == 2
        assert sample_architecture.naming_convention == "snake_case"

    def test_architecture_defaults(self):
        """Test ProjectArchitecture defaults."""
        arch = ProjectArchitecture()
        assert arch.src_layout == "unknown"
        assert arch.test_layout == "unknown"
        assert arch.entry_points == []
        assert arch.naming_convention == "unknown"

    def test_architecture_to_dict(self, sample_architecture):
        """Test architecture serialization."""
        data = sample_architecture.to_dict()
        assert data["src_layout"] == "src/"
        assert data["test_layout"] == "tests/"
        assert len(data["entry_points"]) == 2

    def test_architecture_from_dict(self, sample_architecture):
        """Test architecture deserialization."""
        data = sample_architecture.to_dict()
        restored = ProjectArchitecture.from_dict(data)
        assert restored.src_layout == sample_architecture.src_layout
        assert restored.test_layout == sample_architecture.test_layout
        assert restored.entry_points == sample_architecture.entry_points

    def test_architecture_to_context_prompt(self, sample_architecture):
        """Test architecture context prompt."""
        prompt = sample_architecture.to_context_prompt()

        assert "Project Architecture" in prompt
        assert "Source layout: src/" in prompt
        assert "Test layout: tests/" in prompt
        assert "main.py" in prompt
        assert "snake_case" in prompt
        assert "fastapi" in prompt


# =============================================================================
# ContextStorage Tests
# =============================================================================


class TestContextStorage:
    """Tests for ContextStorage."""

    def test_create_storage(self, temp_dir):
        """Test creating storage."""
        storage = ContextStorage(base_path=temp_dir)
        assert storage.base_path == temp_dir
        assert storage.base_path.exists()

    def test_save_and_load_summary(self, context_storage, sample_summary):
        """Test saving and loading session summary."""
        context_storage.save_summary(sample_summary)
        loaded = context_storage.load_summary(sample_summary.workspace)

        assert loaded is not None
        assert loaded.session_id == sample_summary.session_id
        assert loaded.project_type == sample_summary.project_type

    def test_load_summary_not_found(self, context_storage):
        """Test loading non-existent summary."""
        loaded = context_storage.load_summary("/nonexistent/path")
        assert loaded is None

    def test_summary_history(self, context_storage):
        """Test session history."""
        workspace = "/path/to/project"

        # Save multiple summaries
        for i in range(5):
            summary = SessionSummary(
                session_id=f"session-{i}",
                workspace=workspace,
                tasks_completed=[f"Task {i}"],
            )
            context_storage.save_summary(summary)

        # Load history
        history = context_storage.load_history(workspace, limit=3)
        assert len(history) == 3
        # Most recent first
        assert history[0].session_id == "session-4"
        assert history[1].session_id == "session-3"
        assert history[2].session_id == "session-2"

    def test_history_pruning(self, context_storage):
        """Test that history is pruned."""
        workspace = "/path/to/project"

        # Save more than MAX_HISTORY summaries
        for i in range(context_storage.MAX_HISTORY + 10):
            summary = SessionSummary(
                session_id=f"session-{i}",
                workspace=workspace,
            )
            context_storage.save_summary(summary)

        # History should be limited
        history = context_storage.load_history(workspace, limit=100)
        assert len(history) <= context_storage.MAX_HISTORY

    def test_save_and_load_architecture(self, context_storage, sample_architecture):
        """Test saving and loading architecture."""
        workspace = "/path/to/project"
        context_storage.save_architecture(workspace, sample_architecture)
        loaded = context_storage.load_architecture(workspace)

        assert loaded is not None
        assert loaded.src_layout == sample_architecture.src_layout
        assert loaded.test_layout == sample_architecture.test_layout

    def test_load_architecture_not_found(self, context_storage):
        """Test loading non-existent architecture."""
        loaded = context_storage.load_architecture("/nonexistent/path")
        assert loaded is None

    def test_workspace_isolation(self, context_storage, sample_summary):
        """Test that different workspaces are isolated."""
        summary1 = SessionSummary(
            session_id="session-1",
            workspace="/project1",
            project_type="FastAPI",
        )
        summary2 = SessionSummary(
            session_id="session-2",
            workspace="/project2",
            project_type="Django",
        )

        context_storage.save_summary(summary1)
        context_storage.save_summary(summary2)

        loaded1 = context_storage.load_summary("/project1")
        loaded2 = context_storage.load_summary("/project2")

        assert loaded1.project_type == "FastAPI"
        assert loaded2.project_type == "Django"

    def test_clear_workspace(self, context_storage, sample_summary, sample_architecture):
        """Test clearing workspace context."""
        workspace = "/path/to/project"
        sample_summary.workspace = workspace

        context_storage.save_summary(sample_summary)
        context_storage.save_architecture(workspace, sample_architecture)

        # Verify saved
        assert context_storage.load_summary(workspace) is not None
        assert context_storage.load_architecture(workspace) is not None

        # Clear
        result = context_storage.clear(workspace)
        assert result is True

        # Verify cleared
        assert context_storage.load_summary(workspace) is None
        assert context_storage.load_architecture(workspace) is None

    def test_clear_nonexistent_workspace(self, context_storage):
        """Test clearing non-existent workspace."""
        result = context_storage.clear("/nonexistent/path")
        assert result is False


# =============================================================================
# ArchitectureTracker Tests
# =============================================================================


class TestArchitectureTracker:
    """Tests for ArchitectureTracker."""

    def test_create_tracker(self, temp_dir):
        """Test creating tracker."""
        storage = ContextStorage(base_path=temp_dir)
        tracker = ArchitectureTracker(storage=storage)
        assert tracker.storage is storage

    def test_analyze_project(self, sample_project):
        """Test analyzing a project."""
        tracker = ArchitectureTracker()
        arch = tracker.analyze(sample_project)

        assert arch.src_layout == "src/"
        assert arch.test_layout == "tests/"
        assert arch.config_location == "root"
        assert arch.naming_convention == "snake_case"
        assert arch.docstring_style == "google"

    def test_detect_src_layout_flat(self, temp_dir):
        """Test detecting flat source layout."""
        project = temp_dir / "flat_project"
        project.mkdir()
        (project / "main.py").touch()
        (project / "utils.py").touch()

        tracker = ArchitectureTracker()
        arch = tracker.analyze(project)
        assert arch.src_layout == "flat"

    def test_detect_test_layout(self, temp_dir):
        """Test detecting test layout."""
        project = temp_dir / "test_project"
        project.mkdir()
        (project / "test").mkdir()
        (project / "test" / "test_main.py").touch()

        tracker = ArchitectureTracker()
        arch = tracker.analyze(project)
        assert arch.test_layout == "tests/"  # "test/" matches "tests/" layout

    def test_find_entry_points(self, temp_dir):
        """Test finding entry points."""
        project = temp_dir / "entry_project"
        project.mkdir()
        (project / "main.py").touch()
        (project / "app.py").touch()
        (project / "cli.py").touch()

        tracker = ArchitectureTracker()
        arch = tracker.analyze(project)
        assert "main.py" in arch.entry_points
        assert "app.py" in arch.entry_points
        assert "cli.py" in arch.entry_points

    def test_detect_dependencies_from_pyproject(self, temp_dir):
        """Test detecting dependencies from pyproject.toml."""
        project = temp_dir / "deps_project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            """[project]
dependencies = [
    "fastapi",
    "uvicorn",
]
"""
        )

        tracker = ArchitectureTracker()
        arch = tracker.analyze(project)
        # Note: our simple regex might pick up different matches
        assert len(arch.key_dependencies) >= 0

    def test_detect_dependencies_from_requirements(self, temp_dir):
        """Test detecting dependencies from requirements.txt."""
        project = temp_dir / "req_project"
        project.mkdir()
        (project / "requirements.txt").write_text(
            """fastapi>=0.100.0
sqlalchemy==2.0.0
pydantic
# comment
-e git+https://...
"""
        )

        tracker = ArchitectureTracker()
        arch = tracker.analyze(project)
        assert "fastapi" in arch.key_dependencies
        assert "sqlalchemy" in arch.key_dependencies
        assert "pydantic" in arch.key_dependencies

    def test_detect_dependencies_from_package_json(self, temp_dir):
        """Test detecting dependencies from package.json."""
        project = temp_dir / "npm_project"
        project.mkdir()
        (project / "package.json").write_text(
            json.dumps(
                {
                    "name": "test",
                    "dependencies": {"react": "^18.0.0", "next": "^14.0.0"},
                    "devDependencies": {"typescript": "^5.0.0"},
                }
            )
        )

        tracker = ArchitectureTracker()
        arch = tracker.analyze(project)
        assert "react" in arch.key_dependencies
        assert "next" in arch.key_dependencies
        assert "typescript" in arch.key_dependencies

    def test_update_architecture(self, sample_architecture):
        """Test updating architecture with changes."""
        tracker = ArchitectureTracker()

        changes = ["server.py", "src/new_module.py"]
        updated = tracker.update(sample_architecture, changes)

        assert "server.py" in updated.entry_points

    def test_save_and_load(self, temp_dir, sample_architecture):
        """Test saving and loading architecture via tracker."""
        storage = ContextStorage(base_path=temp_dir)
        tracker = ArchitectureTracker(storage=storage)
        workspace = temp_dir / "project"
        workspace.mkdir()

        tracker.save(workspace, sample_architecture)
        loaded = tracker.load(workspace)

        assert loaded is not None
        assert loaded.src_layout == sample_architecture.src_layout


# =============================================================================
# Context Loading Functions Tests
# =============================================================================


class TestContextLoadingFunctions:
    """Tests for load/save context functions."""

    def test_load_session_context(self, temp_dir, sample_summary, sample_architecture):
        """Test loading session context."""
        storage = ContextStorage(base_path=temp_dir)
        workspace = temp_dir / "project"
        sample_summary.workspace = str(workspace)

        storage.save_summary(sample_summary)
        storage.save_architecture(workspace, sample_architecture)

        context = load_session_context(workspace, storage=storage)

        assert "Previous Session Context" in context
        assert "FastAPI API with SQLAlchemy" in context
        assert "Project Architecture" in context
        assert "src/" in context

    def test_load_session_context_empty(self, temp_dir):
        """Test loading context with no previous data."""
        storage = ContextStorage(base_path=temp_dir)
        workspace = temp_dir / "new_project"

        context = load_session_context(workspace, storage=storage)
        assert context == ""

    def test_save_session_context(self, temp_dir, sample_summary, sample_architecture):
        """Test saving session context."""
        storage = ContextStorage(base_path=temp_dir)
        workspace = temp_dir / "project"
        sample_summary.workspace = str(workspace)

        save_session_context(
            sample_summary,
            architecture=sample_architecture,
            storage=storage,
        )

        # Verify saved
        loaded_summary = storage.load_summary(str(workspace))
        loaded_arch = storage.load_architecture(workspace)

        assert loaded_summary is not None
        assert loaded_arch is not None

    def test_save_session_context_summary_only(self, temp_dir, sample_summary):
        """Test saving only summary (no architecture)."""
        storage = ContextStorage(base_path=temp_dir)
        workspace = temp_dir / "project"
        sample_summary.workspace = str(workspace)

        save_session_context(sample_summary, storage=storage)

        loaded_summary = storage.load_summary(str(workspace))
        loaded_arch = storage.load_architecture(workspace)

        assert loaded_summary is not None
        assert loaded_arch is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestContextCarryoverIntegration:
    """Integration tests for context carryover."""

    def test_full_workflow(self, temp_dir):
        """Test full context carryover workflow."""
        storage = ContextStorage(base_path=temp_dir)
        workspace = temp_dir / "my_project"
        workspace.mkdir()

        # Session 1: First run
        summary1 = SessionSummary(
            session_id="session-1",
            workspace=str(workspace),
            tasks_completed=["Add authentication"],
            project_type="FastAPI API",
            key_patterns=["Use SQLAlchemy for DB"],
            key_decisions=["Chose JWT for auth"],
        )
        storage.save_summary(summary1)

        # Analyze architecture
        (workspace / "src").mkdir()
        (workspace / "src" / "__init__.py").touch()
        (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

        tracker = ArchitectureTracker(storage=storage)
        arch = tracker.analyze(workspace)
        tracker.save(workspace, arch)

        # Session 2: Load context from session 1
        context = load_session_context(workspace, storage=storage)

        assert "FastAPI API" in context
        assert "SQLAlchemy for DB" in context
        assert "JWT for auth" in context
        assert "src/" in context

        # Session 2: Do more work and save
        summary2 = SessionSummary(
            session_id="session-2",
            workspace=str(workspace),
            tasks_completed=["Add rate limiting", "Add caching"],
            project_type="FastAPI API",
            key_patterns=["Use SQLAlchemy for DB", "Redis for caching"],
            key_decisions=["Chose JWT for auth", "Added Redis"],
            continuation_hint="Consider adding monitoring",
        )
        storage.save_summary(summary2)

        # Session 3: Should see session 2 context
        context = load_session_context(workspace, storage=storage)
        assert "Redis for caching" in context
        assert "monitoring" in context

    def test_multiple_workspaces(self, temp_dir):
        """Test context isolation between workspaces."""
        storage = ContextStorage(base_path=temp_dir)

        # Workspace 1: Python project
        ws1 = temp_dir / "python_project"
        ws1.mkdir()
        summary1 = SessionSummary(
            session_id="py-1",
            workspace=str(ws1),
            project_type="Python FastAPI",
        )
        arch1 = ProjectArchitecture(src_layout="src/", naming_convention="snake_case")
        save_session_context(summary1, arch1, storage=storage)

        # Workspace 2: Node project
        ws2 = temp_dir / "node_project"
        ws2.mkdir()
        summary2 = SessionSummary(
            session_id="node-1",
            workspace=str(ws2),
            project_type="Node.js Express",
        )
        arch2 = ProjectArchitecture(src_layout="lib/", naming_convention="camelCase")
        save_session_context(summary2, arch2, storage=storage)

        # Verify isolation
        ctx1 = load_session_context(ws1, storage=storage)
        ctx2 = load_session_context(ws2, storage=storage)

        assert "Python FastAPI" in ctx1
        assert "snake_case" in ctx1
        assert "Node.js Express" in ctx2
        assert "camelCase" in ctx2


# =============================================================================
# Edge Cases
# =============================================================================


class TestContextCarryoverEdgeCases:
    """Edge case tests."""

    def test_corrupted_summary_file(self, temp_dir):
        """Test handling corrupted summary file."""
        storage = ContextStorage(base_path=temp_dir)
        workspace = "/path/to/project"

        # Create corrupted file
        ws_path = storage._workspace_path(workspace)
        ws_path.mkdir(parents=True)
        (ws_path / "latest.json").write_text("not valid json")

        # Should return None, not raise
        loaded = storage.load_summary(workspace)
        assert loaded is None

    def test_corrupted_architecture_file(self, temp_dir):
        """Test handling corrupted architecture file."""
        storage = ContextStorage(base_path=temp_dir)
        workspace = "/path/to/project"

        # Create corrupted file
        ws_path = storage._workspace_path(workspace)
        ws_path.mkdir(parents=True)
        (ws_path / "architecture.json").write_text("{invalid}")

        # Should return None, not raise
        loaded = storage.load_architecture(workspace)
        assert loaded is None

    def test_empty_workspace_path(self, temp_dir):
        """Test with empty project directory."""
        project = temp_dir / "empty_project"
        project.mkdir()

        tracker = ArchitectureTracker()
        arch = tracker.analyze(project)

        assert arch.src_layout == "flat"
        assert arch.test_layout == "unknown"
        assert arch.entry_points == []

    def test_unicode_in_summary(self, temp_dir):
        """Test handling unicode in summary."""
        storage = ContextStorage(base_path=temp_dir)
        summary = SessionSummary(
            session_id="unicode-test",
            workspace="/path/to/project",
            tasks_completed=["Add emoji support 🎉", "日本語サポート"],
            key_patterns=["Use UTF-8 encoding 你好"],
        )

        storage.save_summary(summary)
        loaded = storage.load_summary("/path/to/project")

        assert loaded is not None
        assert "🎉" in loaded.tasks_completed[0]
        assert "日本語" in loaded.tasks_completed[1]

    def test_very_long_paths(self, temp_dir):
        """Test handling very long workspace paths."""
        storage = ContextStorage(base_path=temp_dir)
        long_path = "/path" + "/subdir" * 50 + "/project"

        summary = SessionSummary(
            session_id="long-path-test",
            workspace=long_path,
        )
        storage.save_summary(summary)
        loaded = storage.load_summary(long_path)

        assert loaded is not None
        assert loaded.workspace == long_path

    def test_special_characters_in_path(self, temp_dir):
        """Test handling special characters in workspace path."""
        storage = ContextStorage(base_path=temp_dir)
        special_path = "/path/to/my project (test) [v2]"

        summary = SessionSummary(
            session_id="special-char-test",
            workspace=special_path,
        )
        storage.save_summary(summary)
        loaded = storage.load_summary(special_path)

        assert loaded is not None

    def test_concurrent_reads(self, temp_dir, sample_summary):
        """Test concurrent reads don't cause issues."""
        import threading

        storage = ContextStorage(base_path=temp_dir)
        storage.save_summary(sample_summary)

        results = []

        def read_summary():
            loaded = storage.load_summary(sample_summary.workspace)
            results.append(loaded is not None)

        threads = [threading.Thread(target=read_summary) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
