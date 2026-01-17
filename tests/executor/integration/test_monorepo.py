"""Integration tests for multi-language monorepo execution (Phase 6.2.4).

Tests end-to-end execution of the executor graph with monorepo projects
containing multiple languages:
- Python backend
- TypeScript frontend
- Cross-project task execution

These tests require actual LLM API access and are skipped by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.graph import ExecutorGraph

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def monorepo(tmp_path: Path) -> Path:
    """Create a multi-language monorepo."""
    # Root package.json (for workspace management)
    root_package = {
        "name": "monorepo",
        "private": True,
        "workspaces": ["backend", "frontend"],
    }
    (tmp_path / "package.json").write_text(json.dumps(root_package, indent=2))

    # Python backend
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "pyproject.toml").write_text("""\
[project]
name = "backend"
version = "0.1.0"

[project.optional-dependencies]
dev = ["pytest"]
""")
    (backend / "src").mkdir()
    (backend / "src" / "__init__.py").write_text("")
    (backend / "tests").mkdir()
    (backend / "tests" / "__init__.py").write_text("")

    # TypeScript frontend
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    frontend_package = {
        "name": "frontend",
        "version": "1.0.0",
        "scripts": {
            "build": "tsc",
            "dev": "next dev",
        },
    }
    (frontend / "package.json").write_text(json.dumps(frontend_package, indent=2))
    (frontend / "src").mkdir()
    (frontend / "src" / "components").mkdir()

    return tmp_path


@pytest.fixture
def monorepo_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap for full-stack feature."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Full Stack Feature

## Overview
Add a user profile feature across backend and frontend.

## Tasks

### Phase 1: Backend API

- [ ] **Add user API endpoint**
  - Description: Create API endpoint in backend/src/api/users.py
  - Files: backend/src/api/users.py

### Phase 2: Frontend Component

- [ ] **Add profile component**
  - Description: Create ProfileCard component in frontend/src/components/ProfileCard.tsx
  - Files: frontend/src/components/ProfileCard.tsx
  - Depends: Add user API endpoint
""")
    return roadmap


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.model = "claude-sonnet-4-20250514"
    return agent


# =============================================================================
# Mock Execution Tests (no LLM required)
# =============================================================================


class TestMonorepoSetup:
    """Tests for monorepo fixture setup."""

    def test_project_structure_exists(self, monorepo: Path) -> None:
        """Verify monorepo structure is created correctly."""
        # Root
        assert (monorepo / "package.json").exists()

        # Backend (Python)
        assert (monorepo / "backend" / "pyproject.toml").exists()
        assert (monorepo / "backend" / "src").is_dir()
        assert (monorepo / "backend" / "tests").is_dir()

        # Frontend (TypeScript)
        assert (monorepo / "frontend" / "package.json").exists()
        assert (monorepo / "frontend" / "src").is_dir()
        assert (monorepo / "frontend" / "src" / "components").is_dir()

    def test_backend_pyproject(self, monorepo: Path) -> None:
        """Verify backend pyproject.toml is valid."""
        content = (monorepo / "backend" / "pyproject.toml").read_text()
        assert 'name = "backend"' in content

    def test_frontend_package_json(self, monorepo: Path) -> None:
        """Verify frontend package.json is valid."""
        package = json.loads((monorepo / "frontend" / "package.json").read_text())
        assert package["name"] == "frontend"

    def test_roadmap_structure(self, monorepo_roadmap: Path) -> None:
        """Verify roadmap is created correctly."""
        content = monorepo_roadmap.read_text()
        assert "# Full Stack Feature" in content
        assert "Backend API" in content
        assert "Frontend Component" in content


class TestMonorepoExecutorInitialization:
    """Tests for ExecutorGraph initialization with monorepos."""

    def test_init_with_monorepo(
        self, monorepo: Path, monorepo_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Can initialize with monorepo project."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(monorepo_roadmap),
            shell_workspace=monorepo,
        )

        assert executor.roadmap_path == str(monorepo_roadmap)
        assert executor.shell_workspace == monorepo


class TestMockedMonorepoExecution:
    """Tests using mocked LLM execution for monorepo projects."""

    @pytest.mark.asyncio
    async def test_executor_runs_with_mock(
        self, monorepo: Path, monorepo_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Executor can be called with mocked graph for monorepo project."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(monorepo_roadmap),
            shell_workspace=monorepo,
        )

        # Mock the graph execution
        with patch.object(executor, "graph") as mock_graph:
            mock_result: dict[str, Any] = {
                "status": "completed",
                "tasks_completed_count": 2,
                "tasks_failed_count": 0,
                "files_modified": [
                    "backend/src/api/users.py",
                    "frontend/src/components/ProfileCard.tsx",
                ],
            }
            mock_graph.arun = AsyncMock(return_value=mock_result)

            result = await executor.arun()

            assert result["status"] == "completed"
            assert result["tasks_completed_count"] == 2
            assert len(result["files_modified"]) == 2

    @pytest.mark.asyncio
    async def test_cross_project_files_tracked(
        self, monorepo: Path, monorepo_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Files from both backend and frontend are tracked."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(monorepo_roadmap),
            shell_workspace=monorepo,
        )

        with patch.object(executor, "graph") as mock_graph:
            mock_result: dict[str, Any] = {
                "status": "completed",
                "tasks_completed_count": 2,
                "files_modified": [
                    "backend/src/api/users.py",
                    "frontend/src/components/ProfileCard.tsx",
                ],
            }
            mock_graph.arun = AsyncMock(return_value=mock_result)

            result = await executor.arun()

            # Verify both projects have files
            files = result["files_modified"]
            has_backend = any("backend" in f for f in files)
            has_frontend = any("frontend" in f for f in files)

            assert has_backend
            assert has_frontend


# =============================================================================
# Integration Tests (require LLM API)
# =============================================================================


@pytest.mark.skip(reason="Integration test - requires LLM API")
class TestMonorepoExecution:
    """Integration tests for monorepo execution.

    These tests run actual LLM calls and should only be run manually
    or in environments with LLM API access.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cross_project_execution(self, monorepo: Path) -> None:
        """Should work across multiple projects."""
        roadmap = monorepo / "ROADMAP.md"
        roadmap.write_text("""\
# Full Stack Feature

## Tasks

- [ ] **Add API endpoint**
  - Description: Add user endpoint in backend/
  - Files: backend/src/api/users.py

- [ ] **Add frontend component**
  - Description: Add component to call API
  - Files: frontend/src/components/UserList.tsx
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=monorepo,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"

        # Verify both were modified
        assert len(list((monorepo / "backend").rglob("*.py"))) > 1
        assert len(list((monorepo / "frontend").rglob("*.tsx"))) > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_shared_types(self, monorepo: Path) -> None:
        """Should handle shared types across projects."""
        # Create shared types directory
        shared = monorepo / "shared"
        shared.mkdir()
        (shared / "types").mkdir()

        roadmap = monorepo / "ROADMAP.md"
        roadmap.write_text("""\
# Add Shared Types

## Tasks

- [ ] **Create shared User type**
  - Description: Create shared/types/user.ts with User interface
  - Files: shared/types/user.ts

- [ ] **Use User type in frontend**
  - Description: Import and use User type in frontend
  - Files: frontend/src/types.ts
  - Depends: Create shared User type
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=monorepo,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"
        assert (monorepo / "shared" / "types" / "user.ts").exists()


# =============================================================================
# Verification Helper Tests
# =============================================================================


class TestMonorepoVerificationHelpers:
    """Tests for monorepo verification helpers."""

    def test_count_python_files(self, monorepo: Path) -> None:
        """Can count Python files in backend."""
        # Create some Python files
        api_dir = monorepo / "backend" / "src" / "api"
        api_dir.mkdir(parents=True)
        (api_dir / "__init__.py").write_text("")
        (api_dir / "users.py").write_text("def get_users(): pass")
        (api_dir / "auth.py").write_text("def login(): pass")

        py_files = list((monorepo / "backend").rglob("*.py"))
        assert len(py_files) >= 4  # __init__.py (x2) + users.py + auth.py

    def test_count_typescript_files(self, monorepo: Path) -> None:
        """Can count TypeScript files in frontend."""
        # Create some TypeScript files
        (monorepo / "frontend" / "src" / "components" / "Button.tsx").write_text(
            "export const Button = () => <button />"
        )
        (monorepo / "frontend" / "src" / "components" / "Card.tsx").write_text(
            "export const Card = () => <div />"
        )

        ts_files = list((monorepo / "frontend").rglob("*.tsx"))
        assert len(ts_files) >= 2

    def test_find_project_roots(self, monorepo: Path) -> None:
        """Can identify project roots in monorepo."""
        project_roots = []

        # Find Python projects
        for pyproject in monorepo.rglob("pyproject.toml"):
            project_roots.append(pyproject.parent)

        # Find Node.js projects (non-root)
        for package_json in monorepo.rglob("package.json"):
            if package_json.parent != monorepo:
                project_roots.append(package_json.parent)

        assert len(project_roots) == 2  # backend and frontend
        assert any("backend" in str(p) for p in project_roots)
        assert any("frontend" in str(p) for p in project_roots)
