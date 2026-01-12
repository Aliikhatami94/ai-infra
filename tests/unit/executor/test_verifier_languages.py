"""Unit tests for language-agnostic verification (Phase 5.9.1 & 5.9.3).

Tests for detect_project_type() and get_test_command() functions
that enable multi-language verification in the executor.

Phase 5.9.1: Implementation of project type detection
Phase 5.9.3: Comprehensive test coverage for language-agnostic verification
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai_infra.executor.verifier import (
    ProjectType,
    detect_project_type,
    get_test_command,
)

# =============================================================================
# detect_project_type Tests
# =============================================================================


class TestDetectProjectType:
    """Tests for project type detection from indicator files."""

    def test_python_pyproject(self, tmp_path: Path) -> None:
        """Detects Python from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        assert detect_project_type(tmp_path) == ProjectType.PYTHON

    def test_python_setup_py(self, tmp_path: Path) -> None:
        """Detects Python from setup.py."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")
        assert detect_project_type(tmp_path) == ProjectType.PYTHON

    def test_python_requirements_txt(self, tmp_path: Path) -> None:
        """Detects Python from requirements.txt."""
        (tmp_path / "requirements.txt").write_text("requests>=2.0\n")
        assert detect_project_type(tmp_path) == ProjectType.PYTHON

    def test_nodejs_package_json(self, tmp_path: Path) -> None:
        """Detects Node.js from package.json without tsconfig."""
        pkg = {"name": "test", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert detect_project_type(tmp_path) == ProjectType.NODEJS

    def test_typescript_with_tsconfig(self, tmp_path: Path) -> None:
        """Detects TypeScript when both package.json and tsconfig.json exist."""
        pkg = {"name": "test", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')
        assert detect_project_type(tmp_path) == ProjectType.TYPESCRIPT

    def test_rust_cargo_toml(self, tmp_path: Path) -> None:
        """Detects Rust from Cargo.toml."""
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n')
        assert detect_project_type(tmp_path) == ProjectType.RUST

    def test_go_go_mod(self, tmp_path: Path) -> None:
        """Detects Go from go.mod."""
        (tmp_path / "go.mod").write_text("module example.com/test\n")
        assert detect_project_type(tmp_path) == ProjectType.GO

    def test_makefile_with_test_target(self, tmp_path: Path) -> None:
        """Detects Makefile when it has a test target."""
        (tmp_path / "Makefile").write_text("test:\n\techo 'running tests'\n")
        assert detect_project_type(tmp_path) == ProjectType.MAKEFILE

    def test_makefile_with_spaced_test_target(self, tmp_path: Path) -> None:
        """Detects Makefile with 'test :' format."""
        (tmp_path / "Makefile").write_text("test :\n\techo 'tests'\n")
        assert detect_project_type(tmp_path) == ProjectType.MAKEFILE

    def test_makefile_without_test_target(self, tmp_path: Path) -> None:
        """Does not detect Makefile if no test target exists."""
        (tmp_path / "Makefile").write_text("build:\n\techo 'building'\n")
        assert detect_project_type(tmp_path) == ProjectType.UNKNOWN

    def test_unknown_empty_directory(self, tmp_path: Path) -> None:
        """Returns UNKNOWN for empty directory."""
        assert detect_project_type(tmp_path) == ProjectType.UNKNOWN

    def test_unknown_no_indicators(self, tmp_path: Path) -> None:
        """Returns UNKNOWN when no indicator files exist."""
        (tmp_path / "README.md").write_text("# My Project\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.c").write_text("int main() { return 0; }")
        assert detect_project_type(tmp_path) == ProjectType.UNKNOWN

    def test_priority_python_over_makefile(self, tmp_path: Path) -> None:
        """Python takes priority over Makefile."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
        assert detect_project_type(tmp_path) == ProjectType.PYTHON

    def test_priority_nodejs_over_makefile(self, tmp_path: Path) -> None:
        """Node.js takes priority over Makefile."""
        pkg = {"name": "test", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "Makefile").write_text("test:\n\tnpm test\n")
        assert detect_project_type(tmp_path) == ProjectType.NODEJS


# =============================================================================
# get_test_command Tests
# =============================================================================


class TestGetTestCommand:
    """Tests for getting the appropriate test command."""

    def test_python_returns_pytest(self, tmp_path: Path) -> None:
        """Python projects use pytest."""
        cmd = get_test_command(ProjectType.PYTHON, tmp_path)
        assert cmd is not None
        assert cmd[0] == sys.executable
        assert "-m" in cmd
        assert "pytest" in cmd

    def test_nodejs_with_npm_test_script(self, tmp_path: Path) -> None:
        """Node.js with test script uses npm test."""
        pkg = {"name": "test", "scripts": {"test": "jest"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        cmd = get_test_command(ProjectType.NODEJS, tmp_path)
        assert cmd == ["npm", "test"]

    def test_nodejs_without_test_script_uses_node_test(self, tmp_path: Path) -> None:
        """Node.js without test script falls back to node --test."""
        pkg = {"name": "test", "scripts": {"start": "node index.js"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        cmd = get_test_command(ProjectType.NODEJS, tmp_path)
        assert cmd is not None
        assert cmd[0] == "node"
        assert "--test" in cmd

    def test_nodejs_no_package_no_tests_returns_none(self, tmp_path: Path) -> None:
        """Node.js without package.json or tests directory returns None."""
        cmd = get_test_command(ProjectType.NODEJS, tmp_path)
        assert cmd is None

    def test_typescript_uses_npm_test(self, tmp_path: Path) -> None:
        """TypeScript projects use npm test."""
        cmd = get_test_command(ProjectType.TYPESCRIPT, tmp_path)
        assert cmd == ["npm", "test"]

    def test_rust_uses_cargo_test(self, tmp_path: Path) -> None:
        """Rust projects use cargo test."""
        cmd = get_test_command(ProjectType.RUST, tmp_path)
        assert cmd == ["cargo", "test"]

    def test_go_uses_go_test(self, tmp_path: Path) -> None:
        """Go projects use go test ./..."""
        cmd = get_test_command(ProjectType.GO, tmp_path)
        assert cmd == ["go", "test", "./..."]

    def test_makefile_uses_make_test(self, tmp_path: Path) -> None:
        """Makefile projects use make test."""
        cmd = get_test_command(ProjectType.MAKEFILE, tmp_path)
        assert cmd == ["make", "test"]

    def test_unknown_returns_none(self, tmp_path: Path) -> None:
        """Unknown projects return None."""
        cmd = get_test_command(ProjectType.UNKNOWN, tmp_path)
        assert cmd is None


# =============================================================================
# Integration Tests - TaskVerifier with different project types
# =============================================================================


class TestVerifierLanguageIntegration:
    """Integration tests for verifier with different project types."""

    @pytest.fixture
    def nodejs_workspace(self, tmp_path: Path) -> Path:
        """Create a Node.js workspace with passing tests."""
        pkg = {
            "name": "test-project",
            "version": "1.0.0",
            "scripts": {"test": "node --test tests/"},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg, indent=2))

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "utils.js").write_text(
            "export function greet(name) { return `Hello, ${name}!`; }\n"
        )

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_utils.mjs").write_text(
            """import assert from 'node:assert';
import test from 'node:test';

test('greet works', () => {
    assert.strictEqual(1 + 1, 2);
});
"""
        )
        return tmp_path

    def test_nodejs_project_detected(self, nodejs_workspace: Path) -> None:
        """Node.js project is correctly detected."""
        project_type = detect_project_type(nodejs_workspace)
        assert project_type == ProjectType.NODEJS

    def test_nodejs_test_command_generated(self, nodejs_workspace: Path) -> None:
        """Correct test command is generated for Node.js project."""
        cmd = get_test_command(ProjectType.NODEJS, nodejs_workspace)
        assert cmd == ["npm", "test"]

    @pytest.fixture
    def python_workspace(self, tmp_path: Path) -> Path:
        """Create a Python workspace with passing tests."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\nversion = "0.1.0"\n')

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        (src_dir / "utils.py").write_text(
            "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"
        )

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_utils.py").write_text(
            """def test_greet():
    assert 1 + 1 == 2
"""
        )
        return tmp_path

    def test_python_project_detected(self, python_workspace: Path) -> None:
        """Python project is correctly detected."""
        project_type = detect_project_type(python_workspace)
        assert project_type == ProjectType.PYTHON

    def test_python_test_command_generated(self, python_workspace: Path) -> None:
        """Correct test command is generated for Python project."""
        cmd = get_test_command(ProjectType.PYTHON, python_workspace)
        assert cmd is not None
        assert "pytest" in cmd


# =============================================================================
# Edge Case Tests (Phase 5.9.3)
# =============================================================================


class TestEdgeCases:
    """Edge case tests for robust language detection."""

    def test_empty_package_json(self, tmp_path: Path) -> None:
        """Empty package.json still detects as Node.js."""
        (tmp_path / "package.json").write_text("{}")
        assert detect_project_type(tmp_path) == ProjectType.NODEJS

    def test_malformed_package_json(self, tmp_path: Path) -> None:
        """Malformed package.json falls back gracefully."""
        (tmp_path / "package.json").write_text("not valid json")
        # Should still detect as Node.js based on file existence
        assert detect_project_type(tmp_path) == ProjectType.NODEJS

    def test_empty_pyproject_toml(self, tmp_path: Path) -> None:
        """Empty pyproject.toml still detects as Python."""
        (tmp_path / "pyproject.toml").write_text("")
        assert detect_project_type(tmp_path) == ProjectType.PYTHON

    def test_symlink_to_indicator_file(self, tmp_path: Path) -> None:
        """Symlinks to indicator files are followed."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "pyproject.toml").write_text('[project]\nname = "test"\n')

        link_dir = tmp_path / "linked"
        link_dir.mkdir()
        (link_dir / "pyproject.toml").symlink_to(real_dir / "pyproject.toml")

        assert detect_project_type(link_dir) == ProjectType.PYTHON

    def test_nested_project_uses_root(self, tmp_path: Path) -> None:
        """Detection uses root directory, not nested projects."""
        # Root is Python
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "root"\n')

        # Nested Node.js project should be ignored
        nested = tmp_path / "subproject"
        nested.mkdir()
        (nested / "package.json").write_text('{"name": "nested"}')

        assert detect_project_type(tmp_path) == ProjectType.PYTHON

    def test_get_test_command_with_malformed_package_json(self, tmp_path: Path) -> None:
        """get_test_command handles malformed package.json gracefully."""
        (tmp_path / "package.json").write_text("invalid json {{{}}")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Should fall back to node --test
        cmd = get_test_command(ProjectType.NODEJS, tmp_path)
        assert cmd is not None
        assert cmd[0] == "node"
        assert "--test" in cmd

    def test_all_project_types_have_values(self) -> None:
        """All ProjectType enum values are strings."""
        for project_type in ProjectType:
            assert isinstance(project_type.value, str)
            assert len(project_type.value) > 0

    def test_detection_order_deterministic(self, tmp_path: Path) -> None:
        """Detection order is deterministic when multiple indicators exist."""
        # Python indicators should always win over Makefile
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        (tmp_path / "requirements.txt").write_text("requests\n")
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")

        # Run detection multiple times to ensure consistency
        results = [detect_project_type(tmp_path) for _ in range(5)]
        assert all(r == ProjectType.PYTHON for r in results)

    def test_path_as_string_works(self, tmp_path: Path) -> None:
        """Detection works with string paths, not just Path objects."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        # Pass as string instead of Path
        assert detect_project_type(str(tmp_path)) == ProjectType.PYTHON
