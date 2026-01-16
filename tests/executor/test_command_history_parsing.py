"""Tests for Phase 16.5.2: Command History Parsing Bug Fix.

Tests that _analyze_file_changes() properly handles both string commands
and ShellResult objects in command history.
"""

from __future__ import annotations

import pytest

from ai_infra.executor.agents.coder import CoderAgent
from ai_infra.llm.shell.types import ShellResult

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def coder_agent() -> CoderAgent:
    """Create a CoderAgent instance for testing."""
    return CoderAgent()


@pytest.fixture
def sample_shell_result() -> ShellResult:
    """Create a sample ShellResult."""
    return ShellResult(
        success=True,
        exit_code=0,
        stdout="File created",
        stderr="",
        command="touch new_file.py",
        duration_ms=5.0,
    )


# =============================================================================
# _extract_command_string Tests
# =============================================================================


class TestExtractCommandString:
    """Tests for _extract_command_string helper method."""

    def test_extract_from_string(self, coder_agent: CoderAgent) -> None:
        """String commands should be returned as-is."""
        cmd = "cat > file.py << 'EOF'"
        result = coder_agent._extract_command_string(cmd)
        assert result == cmd

    def test_extract_from_shell_result(self, coder_agent: CoderAgent) -> None:
        """ShellResult.command should be extracted."""
        shell_result = ShellResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            command="echo 'hello' > test.txt",
            duration_ms=1.0,
        )
        result = coder_agent._extract_command_string(shell_result)
        assert result == "echo 'hello' > test.txt"

    def test_extract_from_dict_with_command(self, coder_agent: CoderAgent) -> None:
        """Dict-like objects with 'command' key should work."""

        class DictLike:
            def get(self, key: str, default: str = "") -> str:
                if key == "command":
                    return "ls -la"
                return default

        result = coder_agent._extract_command_string(DictLike())  # type: ignore
        assert result == "ls -la"

    def test_extract_fallback_to_str(self, coder_agent: CoderAgent) -> None:
        """Unknown types should fall back to str()."""
        result = coder_agent._extract_command_string(12345)  # type: ignore
        assert result == "12345"


# =============================================================================
# _analyze_file_changes Tests
# =============================================================================


class TestAnalyzeFileChanges:
    """Tests for _analyze_file_changes with mixed command types."""

    def test_string_commands_only(self, coder_agent: CoderAgent) -> None:
        """Pure string command history should work."""
        commands: list[str | ShellResult] = [
            "touch new_file.py",
            "cat > config.json << 'EOF'\n{}\nEOF",
            "sed -i 's/old/new/g' existing.py",
        ]
        created, modified = coder_agent._analyze_file_changes(commands)

        assert "new_file.py" in created
        assert "config.json" in created
        assert "existing.py" in modified

    def test_shell_result_commands_only(self, coder_agent: CoderAgent) -> None:
        """Pure ShellResult command history should work."""
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="touch app.py",
                duration_ms=1.0,
            ),
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="echo 'data' > output.txt",
                duration_ms=1.0,
            ),
        ]
        created, modified = coder_agent._analyze_file_changes(commands)

        assert "app.py" in created
        assert "output.txt" in created

    def test_mixed_command_history(self, coder_agent: CoderAgent) -> None:
        """Mixed string and ShellResult commands should work."""
        commands: list[str | ShellResult] = [
            "touch string_file.py",  # String command
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="cat > shell_result_file.json << 'EOF'\n{}\nEOF",
                duration_ms=1.0,
            ),
            "echo 'modified' >> existing.txt",  # String command (append)
            ShellResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="error",
                command="sed -i 's/x/y/g' config.yaml",
                duration_ms=2.0,
            ),
        ]
        created, modified = coder_agent._analyze_file_changes(commands)

        assert "string_file.py" in created
        assert "shell_result_file.json" in created
        assert "existing.txt" in modified
        assert "config.yaml" in modified

    def test_cat_heredoc_with_shell_result(self, coder_agent: CoderAgent) -> None:
        """cat > heredoc pattern should be detected in ShellResult."""
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="cat > src/main.py << 'EOF'\nprint('hello')\nEOF",
                duration_ms=5.0,
            ),
        ]
        created, _ = coder_agent._analyze_file_changes(commands)
        assert "src/main.py" in created

    def test_touch_command_in_shell_result(self, coder_agent: CoderAgent) -> None:
        """touch command should be detected in ShellResult."""
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="touch __init__.py",
                duration_ms=1.0,
            ),
        ]
        created, _ = coder_agent._analyze_file_changes(commands)
        assert "__init__.py" in created

    def test_echo_redirect_in_shell_result(self, coder_agent: CoderAgent) -> None:
        """echo redirect should be detected in ShellResult."""
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="echo 'content' > new.txt",
                duration_ms=1.0,
            ),
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="echo 'append' >> existing.txt",
                duration_ms=1.0,
            ),
        ]
        created, modified = coder_agent._analyze_file_changes(commands)

        assert "new.txt" in created
        assert "existing.txt" in modified

    def test_sed_inplace_in_shell_result(self, coder_agent: CoderAgent) -> None:
        """sed -i should be detected in ShellResult."""
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="sed -i 's/old/new/g' path/to/file.py",
                duration_ms=3.0,
            ),
        ]
        _, modified = coder_agent._analyze_file_changes(commands)
        assert "path/to/file.py" in modified

    def test_empty_command_history(self, coder_agent: CoderAgent) -> None:
        """Empty command history should return empty lists."""
        commands: list[str | ShellResult] = []
        created, modified = coder_agent._analyze_file_changes(commands)

        assert created == []
        assert modified == []

    def test_no_file_operations(self, coder_agent: CoderAgent) -> None:
        """Commands without file operations should return empty lists."""
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="hello",
                stderr="",
                command="echo 'hello'",  # No redirect
                duration_ms=1.0,
            ),
            "ls -la",
            "pwd",
            ShellResult(
                success=True,
                exit_code=0,
                stdout="/home/user",
                stderr="",
                command="cd /home/user",
                duration_ms=0.5,
            ),
        ]
        created, modified = coder_agent._analyze_file_changes(commands)

        assert created == []
        assert modified == []


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionPhase16_5_2:
    """Regression tests for the original bug."""

    def test_shell_result_does_not_raise_attribute_error(self, coder_agent: CoderAgent) -> None:
        """ShellResult should not cause AttributeError when calling .lower().

        This was the original bug - command_history contained ShellResult
        objects, and the code tried to call .lower() directly on them.
        """
        # This should NOT raise AttributeError
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="touch test.py",
                duration_ms=1.0,
            ),
        ]

        # The bug was: AttributeError: 'ShellResult' object has no attribute 'lower'
        # After fix, this should work without error
        created, modified = coder_agent._analyze_file_changes(commands)

        assert "test.py" in created

    def test_case_insensitive_matching_still_works(self, coder_agent: CoderAgent) -> None:
        """Case-insensitive detection should work for sed -i and mkdir patterns.

        Note: touch detection uses startswith() which is case-sensitive by design.
        Only patterns using 'in cmd_lower' are case-insensitive.
        """
        commands: list[str | ShellResult] = [
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="touch lowercase.py",  # lowercase touch works
                duration_ms=1.0,
            ),
            ShellResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                command="SED -i 's/x/y/' MixedCase.txt",  # sed -i is case-insensitive
                duration_ms=1.0,
            ),
        ]
        created, modified = coder_agent._analyze_file_changes(commands)

        assert "lowercase.py" in created
        assert "MixedCase.txt" in modified
