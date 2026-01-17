"""Integration tests for Rust project execution (Phase 6.2.3).

Tests end-to-end execution of the executor graph with Rust projects,
including:
- Cargo.toml handling
- Rust code generation
- Compilation verification

These tests require actual LLM API access and are skipped by default.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.graph import ExecutorGraph

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rust_project(tmp_path: Path) -> Path:
    """Create a minimal Rust project."""
    # Cargo.toml
    (tmp_path / "Cargo.toml").write_text("""\
[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
""")

    # Source directory
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text("")

    return tmp_path


@pytest.fixture
def rust_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap for adding Rust code."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Add Function

## Overview
Add a greeting function to the Rust library.

## Tasks

### Phase 1: Implementation

- [ ] **Add greet function**
  - Description: Add a greet function to src/lib.rs that returns "Hello, World!"
  - Files: src/lib.rs

- [ ] **Add unit test**
  - Description: Add a test module with a test for greet()
  - Files: src/lib.rs
  - Depends: Add greet function
""")
    return roadmap


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.model = "claude-sonnet-4-20250514"
    return agent


# =============================================================================
# Helper Functions
# =============================================================================


def cargo_available() -> bool:
    """Check if cargo is available in PATH."""
    return shutil.which("cargo") is not None


# =============================================================================
# Mock Execution Tests (no LLM required)
# =============================================================================


class TestRustProjectSetup:
    """Tests for Rust project fixture setup."""

    def test_project_structure_exists(self, rust_project: Path) -> None:
        """Verify project structure is created correctly."""
        assert (rust_project / "Cargo.toml").exists()
        assert (rust_project / "src").is_dir()
        assert (rust_project / "src" / "lib.rs").exists()

        # Verify Cargo.toml content
        content = (rust_project / "Cargo.toml").read_text()
        assert 'name = "test-project"' in content
        assert 'edition = "2021"' in content

    def test_roadmap_structure(self, rust_roadmap: Path) -> None:
        """Verify roadmap is created correctly."""
        content = rust_roadmap.read_text()
        assert "# Add Function" in content
        assert "greet function" in content


class TestRustExecutorInitialization:
    """Tests for ExecutorGraph initialization with Rust projects."""

    def test_init_with_rust_project(
        self, rust_project: Path, rust_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Can initialize with Rust project."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(rust_roadmap),
            shell_workspace=rust_project,
        )

        assert executor.roadmap_path == str(rust_roadmap)
        assert executor.shell_workspace == rust_project


class TestMockedRustExecution:
    """Tests using mocked LLM execution for Rust projects."""

    @pytest.mark.asyncio
    async def test_executor_runs_with_mock(
        self, rust_project: Path, rust_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Executor can be called with mocked graph for Rust project."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(rust_roadmap),
            shell_workspace=rust_project,
        )

        # Mock the graph execution
        with patch.object(executor, "graph") as mock_graph:
            mock_result: dict[str, Any] = {
                "status": "completed",
                "tasks_completed_count": 2,
                "tasks_failed_count": 0,
                "files_modified": ["src/lib.rs"],
            }
            mock_graph.arun = AsyncMock(return_value=mock_result)

            result = await executor.arun()

            assert result["status"] == "completed"
            assert result["tasks_completed_count"] == 2


# =============================================================================
# Integration Tests (require LLM API and cargo)
# =============================================================================


@pytest.mark.skip(reason="Integration test - requires LLM API")
class TestRustProjectExecution:
    """Integration tests for Rust project execution.

    These tests run actual LLM calls and require cargo to be installed.
    They should only be run manually or in environments with LLM API access.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rust_compilation(self, rust_project: Path) -> None:
        """Should create valid Rust code that compiles."""
        if not cargo_available():
            pytest.skip("cargo not available")

        roadmap = rust_project / "ROADMAP.md"
        roadmap.write_text("""\
# Add Function

## Tasks

- [ ] **Add greet function**
  - Description: Add greet function to lib.rs
  - Files: src/lib.rs
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=rust_project,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"

        # Verify compiles
        import subprocess

        proc = subprocess.run(
            ["cargo", "check"],
            cwd=rust_project,
            capture_output=True,
        )
        assert proc.returncode == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rust_tests_pass(self, rust_project: Path) -> None:
        """Should create Rust code with passing tests."""
        if not cargo_available():
            pytest.skip("cargo not available")

        roadmap = rust_project / "ROADMAP.md"
        roadmap.write_text("""\
# Add Function with Tests

## Tasks

- [ ] **Add greet function with test**
  - Description: Add greet function and unit test to lib.rs
  - Files: src/lib.rs
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=rust_project,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"

        # Verify tests pass
        import subprocess

        proc = subprocess.run(
            ["cargo", "test"],
            cwd=rust_project,
            capture_output=True,
        )
        assert proc.returncode == 0


# =============================================================================
# Verification Helper Tests
# =============================================================================


class TestRustVerificationHelpers:
    """Tests for Rust verification helpers."""

    def test_cargo_toml_valid(self, rust_project: Path) -> None:
        """Cargo.toml can be read as TOML."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[import-not-found]

        content = (rust_project / "Cargo.toml").read_text()
        parsed = tomllib.loads(content)

        assert parsed["package"]["name"] == "test-project"
        assert parsed["package"]["version"] == "0.1.0"

    def test_rust_syntax_in_lib(self, rust_project: Path) -> None:
        """Can write valid Rust syntax to lib.rs."""
        valid_rust = """\
/// Returns a greeting message.
pub fn greet() -> &'static str {
    "Hello, World!"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet(), "Hello, World!");
    }
}
"""
        (rust_project / "src" / "lib.rs").write_text(valid_rust)
        content = (rust_project / "src" / "lib.rs").read_text()

        assert "pub fn greet()" in content
        assert "#[cfg(test)]" in content

    @pytest.mark.skipif(not cargo_available(), reason="cargo not available")
    def test_cargo_check_valid_rust(self, rust_project: Path) -> None:
        """Can verify Rust code compiles with cargo check."""
        import subprocess

        # Write valid Rust code
        (rust_project / "src" / "lib.rs").write_text("""\
pub fn hello() -> &'static str {
    "Hello"
}
""")

        proc = subprocess.run(
            ["cargo", "check"],
            cwd=rust_project,
            capture_output=True,
        )
        assert proc.returncode == 0
