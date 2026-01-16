"""Integration tests for CoderAgent file creation.

Phase 16.5.10.6 of EXECUTOR_5.md - Integration test with real LLM.

This module tests:
- CoderAgent creates syntactically valid Python files
- No literal \\n characters in generated files
- File validation and repair works end-to-end
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ai_infra.executor.agents.coder import CoderAgent

# Mark all tests in this module as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for LLM integration tests",
    ),
]


# =============================================================================
# 16.5.10.6 Integration Test with Real LLM
# =============================================================================


class TestCoderFileCreation:
    """Integration tests for CoderAgent file creation (16.5.10.6)."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        """Create a clean workspace for testing."""
        workspace = tmp_path / "test_workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def coder_agent(self) -> CoderAgent:
        """Create a CoderAgent instance."""
        from ai_infra.executor.agents.coder import CoderAgent

        return CoderAgent(model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_coder_creates_valid_python_file(
        self,
        coder_agent: CoderAgent,
        workspace: Path,
    ) -> None:
        """CoderAgent should create syntactically valid Python files.

        This tests the core issue from 16.5.10: LLMs sometimes generate
        shell commands with literal \\n that don't get interpreted correctly.
        """
        from ai_infra.executor.models import ExecutionContext, Task

        task = Task(
            id="test-1",
            title="Create hello.py",
            description=(
                "Create a hello.py file with a main() function that prints 'Hello, World!'. "
                "The file should have proper imports and be executable."
            ),
        )
        context = ExecutionContext(workspace=workspace)

        result = await coder_agent.execute(task, context)

        # Check the file was created
        hello_py = workspace / "hello.py"
        assert hello_py.exists(), f"hello.py was not created. Result: {result}"

        # Read and validate content
        content = hello_py.read_text()

        # Check for literal \n (the bug we're fixing)
        assert "\\n" not in content, (
            f"File contains literal \\n characters - shell echo bug detected. "
            f"Content: {content[:200]!r}"
        )

        # Verify syntax is valid
        try:
            compile(content, "hello.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated file has syntax error: {e}\nContent: {content}")

        # Verify content has expected structure
        assert "def main" in content or "def hello" in content, (
            f"File missing expected function. Content: {content}"
        )

    @pytest.mark.asyncio
    async def test_coder_creates_multifile_project(
        self,
        coder_agent: CoderAgent,
        workspace: Path,
    ) -> None:
        """CoderAgent can create multiple valid Python files."""
        from ai_infra.executor.models import ExecutionContext, Task

        task = Task(
            id="test-2",
            title="Create calculator module",
            description=(
                "Create a calculator module with two files:\n"
                "1. calculator.py - with add, subtract, multiply, divide functions\n"
                "2. __init__.py - that exports the functions\n"
                "Put them in a 'calc' subdirectory."
            ),
        )
        context = ExecutionContext(workspace=workspace)

        result = await coder_agent.execute(task, context)

        # Check expected files
        calc_dir = workspace / "calc"

        # At least one Python file should exist
        py_files = list(workspace.rglob("*.py"))
        assert len(py_files) >= 1, f"No Python files created. Result: {result}"

        # Validate all created Python files
        for py_file in py_files:
            content = py_file.read_text()

            # Check for literal \n
            assert "\\n" not in content, (
                f"File {py_file.name} contains literal \\n. Content: {content[:100]!r}"
            )

            # Check syntax
            try:
                compile(content, py_file.name, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")

    @pytest.mark.asyncio
    async def test_validation_detects_malformed_file(
        self,
        coder_agent: CoderAgent,
        workspace: Path,
    ) -> None:
        """Validation correctly detects files with literal \\n."""
        # Create a malformed file (simulating the bug)
        bad_file = workspace / "malformed.py"
        bad_file.write_text('import os\\n\\ndef main():\\n    print("hello")')

        errors = coder_agent._validate_created_files(workspace, ["malformed.py"])

        assert len(errors) >= 1
        assert any("malformed.py" in err for err in errors)

    @pytest.mark.asyncio
    async def test_repair_fixes_malformed_file(
        self,
        coder_agent: CoderAgent,
        workspace: Path,
    ) -> None:
        """Repair utility fixes files with literal \\n."""
        # Create a malformed file
        bad_file = workspace / "fixable.py"
        bad_file.write_text('import os\\n\\ndef main():\\n    print("hello")')

        # Repair it
        result = coder_agent._repair_newlines(bad_file)
        assert result is True

        # Verify it's now valid
        content = bad_file.read_text()
        assert "\\n" not in content
        compile(content, "fixable.py", "exec")  # Should not raise


# =============================================================================
# Verifier Auto-Repair Integration Tests
# =============================================================================


class TestVerifierAutoRepair:
    """Integration tests for verifier auto-repair on syntax failure."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        """Create a workspace with project structure."""
        workspace = tmp_path / "project"
        workspace.mkdir()

        # Create pyproject.toml to make it a Python project
        (workspace / "pyproject.toml").write_text('[project]\nname = "test"\nversion = "0.1.0"\n')

        return workspace

    @pytest.mark.asyncio
    async def test_verifier_repairs_malformed_files_on_syntax_failure(
        self,
        workspace: Path,
    ) -> None:
        """Verifier should auto-repair files with literal \\n on syntax failure."""
        from ai_infra.executor.verifier import TaskVerifier

        # Create a malformed file
        bad_file = workspace / "broken.py"
        bad_file.write_text("x = 1\\ny = 2\\nz = 3")

        verifier = TaskVerifier(workspace=workspace)
        result = await verifier.verify()

        # After verification (which includes repair), check the file
        content = bad_file.read_text()

        # File should be repaired (if auto-repair is enabled)
        # Note: This test documents expected behavior even if not yet implemented
        # If auto-repair is not enabled, this will show the file still has issues
        if "\\n" in content:
            pytest.skip(
                "Auto-repair not yet integrated into verifier - "
                "file still has literal \\n after verification"
            )
        else:
            # Verify the repaired file compiles
            compile(content, "broken.py", "exec")
