"""Unit tests for shell environment snapshot capture (Phase 16.1)."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.shell.snapshot import (
    EXCLUDED_ENV_PREFIXES,
    EXCLUDED_ENV_VARS,
    ShellSnapshot,
    ShellType,
    _parse_bash_functions,
    capture_aliases,
    capture_env_vars,
    capture_functions,
    capture_shell_options,
    capture_shell_state,
    detect_shell,
)

# =============================================================================
# ShellType Tests
# =============================================================================


class TestShellType:
    """Tests for ShellType enum."""

    def test_shell_types_exist(self):
        """Test all expected shell types exist."""
        assert ShellType.BASH.value == "bash"
        assert ShellType.ZSH.value == "zsh"
        assert ShellType.FISH.value == "fish"
        assert ShellType.SH.value == "sh"
        assert ShellType.UNKNOWN.value == "unknown"

    def test_shell_type_from_string(self):
        """Test creating ShellType from string."""
        assert ShellType("bash") == ShellType.BASH
        assert ShellType("zsh") == ShellType.ZSH
        assert ShellType("fish") == ShellType.FISH


# =============================================================================
# ShellSnapshot Tests (Phase 16.1.2)
# =============================================================================


class TestShellSnapshot:
    """Tests for ShellSnapshot dataclass (Phase 16.1.2)."""

    def test_default_values(self):
        """Test default snapshot values."""
        snapshot = ShellSnapshot()
        assert snapshot.env_vars == {}
        assert snapshot.aliases == {}
        assert snapshot.functions == {}
        assert snapshot.working_dir == ""
        assert snapshot.shell_options == []
        assert snapshot.shell_type == ShellType.BASH
        assert snapshot.capture_errors == []
        assert isinstance(snapshot.captured_at, datetime)

    def test_custom_values(self):
        """Test snapshot with custom values."""
        now = datetime.now()
        snapshot = ShellSnapshot(
            env_vars={"PATH": "/usr/bin", "HOME": "/home/user"},
            aliases={"ll": "ls -la", "gs": "git status"},
            functions={"greet": "greet() { echo hello; }"},
            working_dir="/home/user/project",
            shell_options=["extglob", "globstar"],
            shell_type=ShellType.ZSH,
            captured_at=now,
            capture_errors=["warning: something"],
        )
        assert snapshot.env_vars == {"PATH": "/usr/bin", "HOME": "/home/user"}
        assert snapshot.aliases == {"ll": "ls -la", "gs": "git status"}
        assert snapshot.functions == {"greet": "greet() { echo hello; }"}
        assert snapshot.working_dir == "/home/user/project"
        assert snapshot.shell_options == ["extglob", "globstar"]
        assert snapshot.shell_type == ShellType.ZSH
        assert snapshot.captured_at == now
        assert snapshot.capture_errors == ["warning: something"]

    def test_to_dict(self):
        """Test to_dict serialization."""
        now = datetime.now()
        snapshot = ShellSnapshot(
            env_vars={"VAR": "value"},
            aliases={"a": "alias"},
            functions={"f": "func"},
            working_dir="/tmp",
            shell_options=["opt"],
            shell_type=ShellType.BASH,
            captured_at=now,
        )
        result = snapshot.to_dict()

        assert result["env_vars"] == {"VAR": "value"}
        assert result["aliases"] == {"a": "alias"}
        assert result["functions"] == {"f": "func"}
        assert result["working_dir"] == "/tmp"
        assert result["shell_options"] == ["opt"]
        assert result["shell_type"] == "bash"
        assert result["captured_at"] == now.isoformat()
        assert result["capture_errors"] == []

    def test_from_dict(self):
        """Test from_dict deserialization."""
        now = datetime.now()
        data = {
            "env_vars": {"KEY": "val"},
            "aliases": {"ll": "ls -l"},
            "functions": {},
            "working_dir": "/home",
            "shell_options": ["opt1", "opt2"],
            "shell_type": "zsh",
            "captured_at": now.isoformat(),
            "capture_errors": [],
        }
        snapshot = ShellSnapshot.from_dict(data)

        assert snapshot.env_vars == {"KEY": "val"}
        assert snapshot.aliases == {"ll": "ls -l"}
        assert snapshot.working_dir == "/home"
        assert snapshot.shell_type == ShellType.ZSH
        assert snapshot.captured_at.isoformat() == now.isoformat()

    def test_count_properties(self):
        """Test count helper properties."""
        snapshot = ShellSnapshot(
            env_vars={"A": "1", "B": "2", "C": "3"},
            aliases={"x": "y", "z": "w"},
            functions={"f1": "def", "f2": "def", "f3": "def", "f4": "def"},
        )
        assert snapshot.env_var_count == 3
        assert snapshot.alias_count == 2
        assert snapshot.function_count == 4

    def test_summary(self):
        """Test summary string generation."""
        snapshot = ShellSnapshot(
            env_vars={"A": "1"},
            aliases={"ll": "ls"},
            functions={},
            working_dir="/home/user",
            shell_type=ShellType.ZSH,
        )
        summary = snapshot.summary()
        assert "zsh" in summary
        assert "1 env vars" in summary
        assert "1 aliases" in summary
        assert "0 functions" in summary
        assert "/home/user" in summary


# =============================================================================
# Shell Detection Tests (Phase 16.1.8)
# =============================================================================


class TestDetectShell:
    """Tests for shell detection (Phase 16.1.8)."""

    def test_detect_from_shell_env_bash(self):
        """Test detection from $SHELL for bash."""
        with patch.dict(os.environ, {"SHELL": "/bin/bash"}, clear=False):
            result = detect_shell()
            assert result == ShellType.BASH

    def test_detect_from_shell_env_zsh(self):
        """Test detection from $SHELL for zsh."""
        with patch.dict(os.environ, {"SHELL": "/usr/local/bin/zsh"}, clear=False):
            result = detect_shell()
            assert result == ShellType.ZSH

    def test_detect_from_shell_env_fish(self):
        """Test detection from $SHELL for fish."""
        with patch.dict(os.environ, {"SHELL": "/usr/bin/fish"}, clear=False):
            result = detect_shell()
            assert result == ShellType.FISH

    def test_detect_unknown_shell(self):
        """Test detection returns UNKNOWN for unrecognized shell."""
        with patch.dict(os.environ, {"SHELL": "/usr/bin/csh"}, clear=False):
            with patch("shutil.which", return_value=None):
                result = detect_shell()
                assert result == ShellType.UNKNOWN


# =============================================================================
# Environment Variable Capture Tests (Phase 16.1.3)
# =============================================================================


class TestCaptureEnvVars:
    """Tests for environment variable capture (Phase 16.1.3)."""

    @pytest.mark.asyncio
    async def test_capture_env_vars_basic(self):
        """Test basic env var capture."""
        with patch.dict(
            os.environ,
            {"TEST_VAR": "test_value", "ANOTHER": "val"},
            clear=True,
        ):
            result = await capture_env_vars(exclude_sensitive=False)

        assert "TEST_VAR" in result
        assert result["TEST_VAR"] == "test_value"

    @pytest.mark.asyncio
    async def test_capture_excludes_sensitive(self):
        """Test sensitive env vars are excluded."""
        with patch.dict(
            os.environ,
            {"NORMAL_VAR": "ok", "OPENAI_API_KEY": "secret123"},
            clear=True,
        ):
            result = await capture_env_vars(exclude_sensitive=True)

        assert "NORMAL_VAR" in result
        assert "OPENAI_API_KEY" not in result

    @pytest.mark.asyncio
    async def test_capture_excludes_prefixes(self):
        """Test env vars with excluded prefixes are filtered."""
        with patch.dict(
            os.environ,
            {"NORMAL": "ok", "BASH_FUNC_test%%": "func", "__INTERNAL": "skip"},
            clear=True,
        ):
            result = await capture_env_vars(exclude_sensitive=True)

        assert "NORMAL" in result
        assert "BASH_FUNC_test%%" not in result
        assert "__INTERNAL" not in result

    @pytest.mark.asyncio
    async def test_capture_with_custom_pattern(self):
        """Test exclude patterns work."""
        with patch.dict(
            os.environ,
            {"KEEP_THIS": "yes", "SKIP_SECRET_VAR": "no"},
            clear=True,
        ):
            result = await capture_env_vars(
                exclude_sensitive=False,
                exclude_patterns=[r"SKIP_.*"],
            )

        assert "KEEP_THIS" in result
        assert "SKIP_SECRET_VAR" not in result


# =============================================================================
# Alias Capture Tests (Phase 16.1.4)
# =============================================================================


class TestCaptureAliases:
    """Tests for alias capture (Phase 16.1.4)."""

    @pytest.mark.asyncio
    async def test_capture_aliases_returns_dict(self):
        """Test alias capture returns a dictionary."""
        # Mock the subprocess to return known aliases
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"alias ll='ls -la'\nalias gs='git status'\n", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await capture_aliases(ShellType.BASH)

        assert isinstance(result, dict)
        assert "ll" in result
        assert result["ll"] == "ls -la"
        assert "gs" in result

    @pytest.mark.asyncio
    async def test_capture_aliases_timeout(self):
        """Test alias capture handles timeout gracefully."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=TimeoutError(),
        ):
            result = await capture_aliases(ShellType.BASH)

        assert result == {}

    @pytest.mark.asyncio
    async def test_capture_aliases_shell_not_found(self):
        """Test alias capture handles missing shell."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError(),
        ):
            result = await capture_aliases(ShellType.BASH)

        assert result == {}


# =============================================================================
# Function Capture Tests (Phase 16.1.5)
# =============================================================================


class TestCaptureFunctions:
    """Tests for function capture (Phase 16.1.5)."""

    def test_parse_bash_functions_simple(self):
        """Test parsing simple bash function."""
        output = """myfunc ()
{
    echo hello
}"""
        result = _parse_bash_functions(output)

        assert "myfunc" in result
        assert "echo hello" in result["myfunc"]

    def test_parse_bash_functions_multiple(self):
        """Test parsing multiple bash functions."""
        output = """func1 ()
{
    echo one
}
func2 ()
{
    echo two
}"""
        result = _parse_bash_functions(output)

        assert "func1" in result
        assert "func2" in result

    def test_parse_bash_functions_nested_braces(self):
        """Test parsing function with nested braces."""
        output = """complex ()
{
    if [ true ]; then
        echo yes
    fi
}"""
        result = _parse_bash_functions(output)

        assert "complex" in result
        assert "if" in result["complex"]

    @pytest.mark.asyncio
    async def test_capture_functions_returns_dict(self):
        """Test function capture returns a dictionary."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"myfunc ()\n{\n    echo test\n}\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await capture_functions(ShellType.BASH)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_capture_functions_timeout(self):
        """Test function capture handles timeout."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=TimeoutError(),
        ):
            result = await capture_functions(ShellType.BASH)

        assert result == {}


# =============================================================================
# Shell Options Capture Tests (Phase 16.1.6)
# =============================================================================


class TestCaptureShellOptions:
    """Tests for shell options capture (Phase 16.1.6)."""

    @pytest.mark.asyncio
    async def test_capture_shell_options_returns_list(self):
        """Test shell options capture returns a list."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"extglob\ton\nglobstar\ton\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await capture_shell_options(ShellType.BASH)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_capture_shell_options_zsh_format(self):
        """Test shell options capture for zsh format."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"autocd\nextendedglob\nnomatch\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await capture_shell_options(ShellType.ZSH)

        assert isinstance(result, list)
        assert "autocd" in result
        assert "extendedglob" in result

    @pytest.mark.asyncio
    async def test_capture_shell_options_timeout(self):
        """Test shell options capture handles timeout."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=TimeoutError(),
        ):
            result = await capture_shell_options(ShellType.BASH)

        assert result == []

    @pytest.mark.asyncio
    async def test_capture_shell_options_unknown_shell(self):
        """Test shell options returns empty for unknown shell."""
        result = await capture_shell_options(ShellType.UNKNOWN)
        assert result == []


# =============================================================================
# Combined Capture Tests (Phase 16.1.7)
# =============================================================================


class TestCaptureShellState:
    """Tests for combined shell state capture (Phase 16.1.7)."""

    @pytest.mark.asyncio
    async def test_capture_shell_state_returns_snapshot(self):
        """Test capture_shell_state returns ShellSnapshot."""
        with patch.dict(os.environ, {"TEST": "value"}, clear=True):
            with patch(
                "ai_infra.executor.shell.snapshot.capture_aliases",
                new_callable=AsyncMock,
                return_value={"ll": "ls -l"},
            ):
                with patch(
                    "ai_infra.executor.shell.snapshot.capture_functions",
                    new_callable=AsyncMock,
                    return_value={},
                ):
                    with patch(
                        "ai_infra.executor.shell.snapshot.capture_shell_options",
                        new_callable=AsyncMock,
                        return_value=["extglob"],
                    ):
                        result = await capture_shell_state(ShellType.BASH)

        assert isinstance(result, ShellSnapshot)
        assert result.shell_type == ShellType.BASH
        assert "TEST" in result.env_vars
        assert "ll" in result.aliases
        assert "extglob" in result.shell_options

    @pytest.mark.asyncio
    async def test_capture_shell_state_auto_detects_shell(self):
        """Test capture_shell_state auto-detects shell type."""
        with patch.dict(os.environ, {"SHELL": "/bin/zsh", "TEST": "val"}, clear=True):
            with patch(
                "ai_infra.executor.shell.snapshot.capture_aliases",
                new_callable=AsyncMock,
                return_value={},
            ):
                with patch(
                    "ai_infra.executor.shell.snapshot.capture_functions",
                    new_callable=AsyncMock,
                    return_value={},
                ):
                    with patch(
                        "ai_infra.executor.shell.snapshot.capture_shell_options",
                        new_callable=AsyncMock,
                        return_value=[],
                    ):
                        result = await capture_shell_state()

        assert result.shell_type == ShellType.ZSH

    @pytest.mark.asyncio
    async def test_capture_shell_state_captures_working_dir(self):
        """Test capture_shell_state includes working directory."""
        cwd = os.getcwd()
        with patch.dict(os.environ, {"TEST": "val"}, clear=True):
            with patch(
                "ai_infra.executor.shell.snapshot.capture_aliases",
                new_callable=AsyncMock,
                return_value={},
            ):
                with patch(
                    "ai_infra.executor.shell.snapshot.capture_functions",
                    new_callable=AsyncMock,
                    return_value={},
                ):
                    with patch(
                        "ai_infra.executor.shell.snapshot.capture_shell_options",
                        new_callable=AsyncMock,
                        return_value=[],
                    ):
                        result = await capture_shell_state(ShellType.BASH)

        assert result.working_dir == cwd

    @pytest.mark.asyncio
    async def test_capture_shell_state_handles_errors(self):
        """Test capture_shell_state records errors but continues."""
        with patch.dict(os.environ, {"TEST": "val"}, clear=True):
            with patch(
                "ai_infra.executor.shell.snapshot.capture_aliases",
                new_callable=AsyncMock,
                side_effect=Exception("alias error"),
            ):
                with patch(
                    "ai_infra.executor.shell.snapshot.capture_functions",
                    new_callable=AsyncMock,
                    return_value={},
                ):
                    with patch(
                        "ai_infra.executor.shell.snapshot.capture_shell_options",
                        new_callable=AsyncMock,
                        return_value=[],
                    ):
                        result = await capture_shell_state(ShellType.BASH)

        assert isinstance(result, ShellSnapshot)
        assert len(result.capture_errors) > 0
        assert "alias error" in result.capture_errors[0]

    @pytest.mark.asyncio
    async def test_capture_shell_state_skip_functions(self):
        """Test capture_shell_state can skip function capture."""
        with patch.dict(os.environ, {"TEST": "val"}, clear=True):
            with patch(
                "ai_infra.executor.shell.snapshot.capture_aliases",
                new_callable=AsyncMock,
                return_value={},
            ):
                with patch(
                    "ai_infra.executor.shell.snapshot.capture_functions",
                    new_callable=AsyncMock,
                    return_value={"should_not_appear": "def"},
                ) as mock_funcs:
                    with patch(
                        "ai_infra.executor.shell.snapshot.capture_shell_options",
                        new_callable=AsyncMock,
                        return_value=[],
                    ):
                        result = await capture_shell_state(
                            ShellType.BASH,
                            capture_functions_flag=False,
                        )

        # Functions should not be captured
        mock_funcs.assert_not_called()
        assert result.functions == {}

    @pytest.mark.asyncio
    async def test_capture_shell_state_skip_aliases(self):
        """Test capture_shell_state can skip alias capture."""
        with patch.dict(os.environ, {"TEST": "val"}, clear=True):
            with patch(
                "ai_infra.executor.shell.snapshot.capture_aliases",
                new_callable=AsyncMock,
                return_value={"should_not_appear": "alias"},
            ) as mock_aliases:
                with patch(
                    "ai_infra.executor.shell.snapshot.capture_functions",
                    new_callable=AsyncMock,
                    return_value={},
                ):
                    with patch(
                        "ai_infra.executor.shell.snapshot.capture_shell_options",
                        new_callable=AsyncMock,
                        return_value=[],
                    ):
                        result = await capture_shell_state(
                            ShellType.BASH,
                            capture_aliases_flag=False,
                        )

        mock_aliases.assert_not_called()
        assert result.aliases == {}


# =============================================================================
# Excluded Variables Tests
# =============================================================================


class TestExcludedVariables:
    """Tests for excluded environment variables."""

    def test_sensitive_vars_in_exclusion_list(self):
        """Test sensitive variables are in exclusion list."""
        assert "OPENAI_API_KEY" in EXCLUDED_ENV_VARS
        assert "ANTHROPIC_API_KEY" in EXCLUDED_ENV_VARS
        assert "AWS_SECRET_ACCESS_KEY" in EXCLUDED_ENV_VARS
        assert "GITHUB_TOKEN" in EXCLUDED_ENV_VARS

    def test_exclusion_prefixes_defined(self):
        """Test exclusion prefixes are defined."""
        assert "BASH_FUNC_" in EXCLUDED_ENV_PREFIXES
        assert "__" in EXCLUDED_ENV_PREFIXES
