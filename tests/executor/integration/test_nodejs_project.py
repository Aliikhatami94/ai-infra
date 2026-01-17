"""Integration tests for Node.js project execution (Phase 6.2.2).

Tests end-to-end execution of the executor graph with Node.js/TypeScript projects,
including:
- TypeScript configuration
- TypeScript file creation
- npm/package.json handling

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
def nodejs_project(tmp_path: Path) -> Path:
    """Create a minimal Node.js project."""
    # package.json
    package = {
        "name": "test-project",
        "version": "1.0.0",
        "description": "Test Node.js project",
        "main": "src/index.js",
        "scripts": {
            "build": "tsc",
            "test": "jest",
        },
        "devDependencies": {},
    }
    (tmp_path / "package.json").write_text(json.dumps(package, indent=2))

    # Source directory
    src = tmp_path / "src"
    src.mkdir()

    return tmp_path


@pytest.fixture
def typescript_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap for adding TypeScript."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Add TypeScript

## Overview
Set up TypeScript for the project.

## Tasks

### Phase 1: Configuration

- [ ] **Add tsconfig.json**
  - Description: Create TypeScript configuration file
  - Files: tsconfig.json

- [ ] **Create index.ts**
  - Description: Create src/index.ts with a hello function
  - Files: src/index.ts
  - Depends: Add tsconfig.json
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


class TestNodeProjectSetup:
    """Tests for Node.js project fixture setup."""

    def test_project_structure_exists(self, nodejs_project: Path) -> None:
        """Verify project structure is created correctly."""
        assert (nodejs_project / "package.json").exists()
        assert (nodejs_project / "src").is_dir()

        # Verify package.json content
        package = json.loads((nodejs_project / "package.json").read_text())
        assert package["name"] == "test-project"
        assert package["version"] == "1.0.0"

    def test_roadmap_structure(self, typescript_roadmap: Path) -> None:
        """Verify roadmap is created correctly."""
        content = typescript_roadmap.read_text()
        assert "# Add TypeScript" in content
        assert "tsconfig.json" in content
        assert "index.ts" in content


class TestNodeExecutorInitialization:
    """Tests for ExecutorGraph initialization with Node.js projects."""

    def test_init_with_node_project(
        self, nodejs_project: Path, typescript_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Can initialize with Node.js project."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(typescript_roadmap),
            shell_workspace=nodejs_project,
        )

        assert executor.roadmap_path == str(typescript_roadmap)
        assert executor.shell_workspace == nodejs_project


class TestMockedNodeExecution:
    """Tests using mocked LLM execution for Node.js projects."""

    @pytest.mark.asyncio
    async def test_executor_runs_with_mock(
        self, nodejs_project: Path, typescript_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Executor can be called with mocked graph for Node.js project."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(typescript_roadmap),
            shell_workspace=nodejs_project,
        )

        # Mock the graph execution
        with patch.object(executor, "graph") as mock_graph:
            mock_result: dict[str, Any] = {
                "status": "completed",
                "tasks_completed_count": 2,
                "tasks_failed_count": 0,
                "files_modified": ["tsconfig.json", "src/index.ts"],
            }
            mock_graph.arun = AsyncMock(return_value=mock_result)

            result = await executor.arun()

            assert result["status"] == "completed"
            assert result["tasks_completed_count"] == 2


# =============================================================================
# Integration Tests (require LLM API)
# =============================================================================


@pytest.mark.skip(reason="Integration test - requires LLM API")
class TestNodeProjectExecution:
    """Integration tests for Node.js project execution.

    These tests run actual LLM calls and should only be run manually
    or in environments with LLM API access.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_creates_typescript_files(self, nodejs_project: Path) -> None:
        """Should create TypeScript files correctly."""
        roadmap = nodejs_project / "ROADMAP.md"
        roadmap.write_text("""\
# Add TypeScript

## Tasks

- [ ] **Add tsconfig.json**
  - Description: Create TypeScript configuration
  - Files: tsconfig.json

- [ ] **Create index.ts**
  - Description: Create src/index.ts with hello function
  - Files: src/index.ts
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=nodejs_project,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"
        assert (nodejs_project / "tsconfig.json").exists()
        assert (nodejs_project / "src" / "index.ts").exists()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_creates_react_component(self, nodejs_project: Path) -> None:
        """Should create React components correctly."""
        # Setup as React project
        (nodejs_project / "src" / "components").mkdir()

        roadmap = nodejs_project / "ROADMAP.md"
        roadmap.write_text("""\
# Add React Component

## Tasks

- [ ] **Create Button component**
  - Description: Create a reusable Button component in src/components/Button.tsx
  - Files: src/components/Button.tsx
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=nodejs_project,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"
        assert (nodejs_project / "src" / "components" / "Button.tsx").exists()


# =============================================================================
# Verification Helper Tests
# =============================================================================


class TestNodeVerificationHelpers:
    """Tests for Node.js verification helpers."""

    def test_json_valid(self) -> None:
        """Can validate JSON syntax."""
        valid_json = '{"name": "test", "version": "1.0.0"}'
        parsed = json.loads(valid_json)
        assert parsed["name"] == "test"

    def test_json_invalid(self) -> None:
        """Can detect JSON syntax errors."""
        invalid_json = '{"name": test}'  # Missing quotes around value
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_tsconfig_structure(self, nodejs_project: Path) -> None:
        """Can create valid tsconfig.json structure."""
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "strict": True,
                "outDir": "./dist",
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules"],
        }

        config_path = nodejs_project / "tsconfig.json"
        config_path.write_text(json.dumps(tsconfig, indent=2))

        # Verify it's valid JSON
        loaded = json.loads(config_path.read_text())
        assert loaded["compilerOptions"]["target"] == "ES2020"
