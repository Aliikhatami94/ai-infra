"""Unit tests for shell environment restoration (Phase 16.3)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ai_infra.executor.shell.restoration import (
    CAUTION_ENV_VARS,
    PROTECTED_ENV_VARS,
    RestorationResult,
    SnapshotDiff,
    diff_snapshots,
    generate_restore_script,
    is_caution_env_var,
    is_protected_env_var,
    restore_aliases,
    restore_env_vars,
    restore_functions,
    restore_shell_state,
    restore_working_dir,
    validate_env_var_value,
)
from ai_infra.executor.shell.snapshot import ShellSnapshot, ShellType

# =============================================================================
# RestorationResult Tests
# =============================================================================


class TestRestorationResult:
    """Tests for RestorationResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = RestorationResult()
        assert result.success is True
        assert result.env_vars_restored == 0
        assert result.env_vars_skipped == 0
        assert result.errors == []
        assert result.warnings == []

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = RestorationResult(
            success=True,
            env_vars_restored=5,
            env_vars_skipped=2,
            aliases_restored=3,
            errors=["error1"],
            warnings=["warn1"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["env_vars_restored"] == 5
        assert d["env_vars_skipped"] == 2
        assert d["aliases_restored"] == 3
        assert d["errors"] == ["error1"]

    def test_summary(self):
        """Test summary generation."""
        result = RestorationResult(
            success=True,
            env_vars_restored=10,
            aliases_restored=5,
            working_dir_restored=True,
        )
        summary = result.summary()
        assert "OK" in summary
        assert "10 env vars" in summary
        assert "5 aliases" in summary
        assert "cwd" in summary

    def test_summary_failed(self):
        """Test summary for failed restoration."""
        result = RestorationResult(success=False)
        summary = result.summary()
        assert "FAILED" in summary


# =============================================================================
# Safety Check Tests (Phase 16.3.7)
# =============================================================================


class TestSafetyChecks:
    """Tests for safety checks (Phase 16.3.7)."""

    def test_protected_env_vars_defined(self):
        """Test protected env vars are defined."""
        assert "PATH" in PROTECTED_ENV_VARS
        assert "HOME" in PROTECTED_ENV_VARS
        assert "USER" in PROTECTED_ENV_VARS
        assert "LD_LIBRARY_PATH" in PROTECTED_ENV_VARS

    def test_caution_env_vars_defined(self):
        """Test caution env vars are defined."""
        assert "PYTHONPATH" in CAUTION_ENV_VARS
        assert "VIRTUAL_ENV" in CAUTION_ENV_VARS

    def test_is_protected_env_var(self):
        """Test is_protected_env_var function."""
        assert is_protected_env_var("PATH") is True
        assert is_protected_env_var("HOME") is True
        assert is_protected_env_var("MY_CUSTOM_VAR") is False

    def test_is_caution_env_var(self):
        """Test is_caution_env_var function."""
        assert is_caution_env_var("PYTHONPATH") is True
        assert is_caution_env_var("MY_CUSTOM_VAR") is False

    def test_validate_env_var_safe_value(self):
        """Test validation of safe value."""
        is_safe, warning = validate_env_var_value("TEST", "normal_value")
        assert is_safe is True
        assert warning is None

    def test_validate_env_var_command_substitution(self):
        """Test detection of command substitution."""
        is_safe, warning = validate_env_var_value("TEST", "$(rm -rf /)")
        assert is_safe is False
        assert "dangerous" in warning.lower()

    def test_validate_env_var_backtick(self):
        """Test detection of backtick command."""
        is_safe, warning = validate_env_var_value("TEST", "`whoami`")
        assert is_safe is False

    def test_validate_env_var_command_chain(self):
        """Test detection of command chaining."""
        is_safe, warning = validate_env_var_value("TEST", "value && rm file")
        assert is_safe is False

    def test_validate_env_var_long_value(self):
        """Test detection of excessively long value."""
        long_value = "x" * 20000
        is_safe, warning = validate_env_var_value("TEST", long_value)
        assert is_safe is False
        assert "length" in warning.lower()


# =============================================================================
# Restore Environment Variables Tests (Phase 16.3.1)
# =============================================================================


class TestRestoreEnvVars:
    """Tests for restore_env_vars (Phase 16.3.1)."""

    def test_restore_basic(self):
        """Test basic env var restoration."""
        snapshot = ShellSnapshot(env_vars={"TEST_VAR_1": "value1", "TEST_VAR_2": "value2"})

        # Clean up any existing test vars
        for key in ["TEST_VAR_1", "TEST_VAR_2"]:
            os.environ.pop(key, None)

        result = restore_env_vars(snapshot)

        assert result.success is True
        assert result.env_vars_restored == 2
        assert os.environ.get("TEST_VAR_1") == "value1"
        assert os.environ.get("TEST_VAR_2") == "value2"

        # Cleanup
        os.environ.pop("TEST_VAR_1", None)
        os.environ.pop("TEST_VAR_2", None)

    def test_restore_skips_protected(self):
        """Test protected vars are skipped."""
        snapshot = ShellSnapshot(env_vars={"PATH": "/bad/path", "TEST_SAFE": "safe_value"})

        original_path = os.environ.get("PATH")
        result = restore_env_vars(snapshot, skip_protected=True)

        assert result.env_vars_skipped >= 1
        assert os.environ.get("PATH") == original_path  # Unchanged
        assert "Skipped protected" in result.warnings[0]

        # Cleanup
        os.environ.pop("TEST_SAFE", None)

    def test_restore_dry_run(self):
        """Test dry run doesn't modify env."""
        snapshot = ShellSnapshot(env_vars={"DRY_RUN_TEST": "value"})

        os.environ.pop("DRY_RUN_TEST", None)
        result = restore_env_vars(snapshot, dry_run=True)

        assert result.env_vars_restored == 1
        assert "DRY_RUN_TEST" not in os.environ

    def test_restore_validates_unsafe(self):
        """Test unsafe values are skipped."""
        snapshot = ShellSnapshot(env_vars={"UNSAFE": "$(evil_command)"})

        result = restore_env_vars(snapshot, validate_values=True)

        assert result.env_vars_skipped == 1
        assert "UNSAFE" not in os.environ


# =============================================================================
# Restore Working Directory Tests (Phase 16.3.4)
# =============================================================================


class TestRestoreWorkingDir:
    """Tests for restore_working_dir (Phase 16.3.4)."""

    def test_restore_to_existing_dir(self):
        """Test restoration to existing directory."""
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Resolve symlinks for macOS compatibility
                tmpdir_resolved = os.path.realpath(tmpdir)
                snapshot = ShellSnapshot(working_dir=tmpdir_resolved)

                result = restore_working_dir(snapshot)

                assert result.success is True
                assert result.working_dir_restored is True
                assert os.path.realpath(os.getcwd()) == tmpdir_resolved
        finally:
            # Always restore original
            os.chdir(original_cwd)

    def test_restore_nonexistent_fails(self):
        """Test restoration to non-existent directory fails."""
        snapshot = ShellSnapshot(working_dir="/nonexistent/path/xyz")

        result = restore_working_dir(snapshot, create_if_missing=False)

        assert result.success is False
        assert result.working_dir_restored is False
        assert "does not exist" in result.errors[0]

    def test_restore_creates_missing(self):
        """Test creating missing directory."""
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                new_dir = Path(tmpdir) / "new" / "nested" / "dir"
                snapshot = ShellSnapshot(working_dir=str(new_dir))

                result = restore_working_dir(snapshot, create_if_missing=True)

                assert result.success is True
                assert result.working_dir_restored is True
                assert new_dir.exists()
        finally:
            os.chdir(original_cwd)

    def test_restore_dry_run(self):
        """Test dry run doesn't change directory."""
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = ShellSnapshot(working_dir=tmpdir)

            result = restore_working_dir(snapshot, dry_run=True)

            assert result.working_dir_restored is True
            assert os.getcwd() == original_cwd  # Unchanged

    def test_restore_empty_working_dir(self):
        """Test handling empty working directory."""
        snapshot = ShellSnapshot(working_dir="")

        result = restore_working_dir(snapshot)

        assert "No working directory" in result.warnings[0]


# =============================================================================
# Restore Aliases Tests (Phase 16.3.2)
# =============================================================================


class TestRestoreAliases:
    """Tests for restore_aliases (Phase 16.3.2)."""

    def test_restore_generates_bash_script(self):
        """Test alias restoration generates bash script."""
        snapshot = ShellSnapshot(
            aliases={"ll": "ls -la", "gs": "git status"},
            shell_type=ShellType.BASH,
        )

        script, result = restore_aliases(snapshot)

        assert result.aliases_restored == 2
        assert "alias ll=" in script
        assert "alias gs=" in script

    def test_restore_generates_fish_script(self):
        """Test alias restoration generates fish script."""
        snapshot = ShellSnapshot(
            aliases={"ll": "ls -la"},
            shell_type=ShellType.FISH,
        )

        script, result = restore_aliases(snapshot)

        assert result.aliases_restored == 1
        assert "alias ll " in script  # Fish syntax


# =============================================================================
# Restore Functions Tests (Phase 16.3.3)
# =============================================================================


class TestRestoreFunctions:
    """Tests for restore_functions (Phase 16.3.3)."""

    def test_restore_generates_script(self):
        """Test function restoration generates script."""
        snapshot = ShellSnapshot(
            functions={
                "greet": "greet() {\n    echo hello\n}",
                "farewell": "farewell() {\n    echo goodbye\n}",
            }
        )

        script, result = restore_functions(snapshot)

        assert result.functions_restored == 2
        assert "greet()" in script
        assert "farewell()" in script


# =============================================================================
# Generate Restore Script Tests (Phase 16.3.6)
# =============================================================================


class TestGenerateRestoreScript:
    """Tests for generate_restore_script (Phase 16.3.6)."""

    def test_generate_basic_script(self):
        """Test basic script generation."""
        snapshot = ShellSnapshot(
            env_vars={"VAR1": "value1"},
            aliases={"ll": "ls -la"},
            functions={"f": "f() { echo hi; }"},
            working_dir="/home/user",
            shell_type=ShellType.BASH,
        )

        script, result = generate_restore_script(snapshot)

        assert "#!/usr/bin/env bash" in script
        assert "export VAR1=" in script
        assert "alias ll=" in script
        assert "f()" in script
        assert "cd " in script

    def test_generate_fish_script(self):
        """Test fish script generation."""
        snapshot = ShellSnapshot(
            env_vars={"VAR1": "value1"},
            shell_type=ShellType.FISH,
        )

        script, result = generate_restore_script(snapshot)

        assert "#!/usr/bin/env fish" in script
        assert "set -gx VAR1" in script

    def test_generate_skips_protected(self):
        """Test script skips protected vars."""
        snapshot = ShellSnapshot(
            env_vars={"PATH": "/bad", "SAFE_VAR": "ok"},
            shell_type=ShellType.BASH,
        )

        script, result = generate_restore_script(snapshot, skip_protected=True)

        assert result.env_vars_skipped >= 1
        assert "Skipped protected: PATH" in script
        assert "export SAFE_VAR=" in script

    def test_generate_writes_file(self):
        """Test script writes to file."""
        snapshot = ShellSnapshot(
            env_vars={"TEST": "value"},
            shell_type=ShellType.BASH,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "restore.sh"
            script, result = generate_restore_script(snapshot, output_path=output_path)

            assert result.script_generated is True
            assert result.script_path == str(output_path)
            assert output_path.exists()

            # Check file is executable
            assert os.access(output_path, os.X_OK)

    def test_generate_selective_includes(self):
        """Test selective inclusion of components."""
        snapshot = ShellSnapshot(
            env_vars={"VAR": "val"},
            aliases={"a": "alias"},
            functions={"f": "func"},
            working_dir="/home",
            shell_type=ShellType.BASH,
        )

        script, result = generate_restore_script(
            snapshot,
            include_env_vars=False,
            include_aliases=True,
            include_functions=False,
            include_working_dir=False,
        )

        assert "export VAR" not in script
        assert "alias a=" in script
        assert "cd " not in script
        assert result.env_vars_restored == 0
        assert result.aliases_restored == 1


# =============================================================================
# Restore Shell State Tests (Phase 16.3.5)
# =============================================================================


class TestRestoreShellState:
    """Tests for restore_shell_state (Phase 16.3.5)."""

    def test_restore_full_state(self):
        """Test full state restoration."""
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_resolved = os.path.realpath(tmpdir)
                snapshot = ShellSnapshot(
                    env_vars={"FULL_RESTORE_TEST": "value"},
                    aliases={"ll": "ls -la"},
                    working_dir=tmpdir_resolved,
                    shell_type=ShellType.BASH,
                )

                result = restore_shell_state(snapshot, generate_script=False)

                assert result.success is True
                assert result.env_vars_restored >= 1
                assert os.environ.get("FULL_RESTORE_TEST") == "value"
                assert os.path.realpath(os.getcwd()) == tmpdir_resolved
        finally:
            # Cleanup
            os.chdir(original_cwd)
            os.environ.pop("FULL_RESTORE_TEST", None)

    def test_restore_with_script_generation(self):
        """Test restoration with script generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "restore.sh"
            snapshot = ShellSnapshot(
                aliases={"test_alias": "echo test"},
                functions={"test_func": "test_func() { echo hi; }"},
                shell_type=ShellType.BASH,
            )

            result = restore_shell_state(
                snapshot,
                restore_env=False,
                restore_cwd=False,
                generate_script=True,
                script_path=script_path,
            )

            assert result.script_generated is True
            assert script_path.exists()

    def test_restore_dry_run(self):
        """Test dry run restoration."""
        original_cwd = os.getcwd()
        snapshot = ShellSnapshot(
            env_vars={"DRY_RUN_FULL": "value"},
            working_dir=os.path.realpath("/tmp"),
            shell_type=ShellType.BASH,
        )

        restore_shell_state(snapshot, dry_run=True)

        assert "DRY_RUN_FULL" not in os.environ
        assert os.getcwd() == original_cwd


# =============================================================================
# Snapshot Diff Tests
# =============================================================================


class TestSnapshotDiff:
    """Tests for SnapshotDiff and diff_snapshots."""

    def test_diff_default_values(self):
        """Test diff default values."""
        diff = SnapshotDiff()
        assert diff.has_changes is False
        assert diff.env_vars_added == {}

    def test_diff_no_changes(self):
        """Test diff with identical snapshots."""
        snap1 = ShellSnapshot(env_vars={"A": "1"}, aliases={"ll": "ls"})
        snap2 = ShellSnapshot(env_vars={"A": "1"}, aliases={"ll": "ls"})

        diff = diff_snapshots(snap1, snap2)

        assert diff.has_changes is False
        assert diff.summary() == "No changes"

    def test_diff_env_vars_added(self):
        """Test detecting added env vars."""
        snap1 = ShellSnapshot(env_vars={"A": "1"})
        snap2 = ShellSnapshot(env_vars={"A": "1", "B": "2"})

        diff = diff_snapshots(snap1, snap2)

        assert diff.has_changes is True
        assert "B" in diff.env_vars_added
        assert diff.env_vars_added["B"] == "2"

    def test_diff_env_vars_removed(self):
        """Test detecting removed env vars."""
        snap1 = ShellSnapshot(env_vars={"A": "1", "B": "2"})
        snap2 = ShellSnapshot(env_vars={"A": "1"})

        diff = diff_snapshots(snap1, snap2)

        assert diff.has_changes is True
        assert "B" in diff.env_vars_removed

    def test_diff_env_vars_changed(self):
        """Test detecting changed env vars."""
        snap1 = ShellSnapshot(env_vars={"A": "old"})
        snap2 = ShellSnapshot(env_vars={"A": "new"})

        diff = diff_snapshots(snap1, snap2)

        assert diff.has_changes is True
        assert "A" in diff.env_vars_changed
        assert diff.env_vars_changed["A"] == ("old", "new")

    def test_diff_aliases(self):
        """Test detecting alias changes."""
        snap1 = ShellSnapshot(aliases={"ll": "ls -l", "old": "cmd"})
        snap2 = ShellSnapshot(aliases={"ll": "ls -la", "new": "cmd2"})

        diff = diff_snapshots(snap1, snap2)

        assert "new" in diff.aliases_added
        assert "old" in diff.aliases_removed
        assert "ll" in diff.aliases_changed

    def test_diff_functions(self):
        """Test detecting function changes."""
        snap1 = ShellSnapshot(functions={"f1": "old_def"})
        snap2 = ShellSnapshot(functions={"f1": "new_def", "f2": "def2"})

        diff = diff_snapshots(snap1, snap2)

        assert "f2" in diff.functions_added
        assert "f1" in diff.functions_changed

    def test_diff_working_dir(self):
        """Test detecting working directory change."""
        snap1 = ShellSnapshot(working_dir="/old/path")
        snap2 = ShellSnapshot(working_dir="/new/path")

        diff = diff_snapshots(snap1, snap2)

        assert diff.working_dir_changed is True
        assert diff.old_working_dir == "/old/path"
        assert diff.new_working_dir == "/new/path"

    def test_diff_summary(self):
        """Test diff summary generation."""
        snap1 = ShellSnapshot(
            env_vars={"A": "1"},
            aliases={"ll": "ls"},
            working_dir="/old",
        )
        snap2 = ShellSnapshot(
            env_vars={"A": "1", "B": "2"},
            aliases={"ll": "ls -la"},
            working_dir="/new",
        )

        diff = diff_snapshots(snap1, snap2)
        summary = diff.summary()

        assert "env:" in summary
        assert "aliases:" in summary
        assert "cwd:" in summary

    def test_diff_to_dict(self):
        """Test diff to_dict serialization."""
        snap1 = ShellSnapshot(env_vars={"A": "old"})
        snap2 = ShellSnapshot(env_vars={"A": "new", "B": "added"})

        diff = diff_snapshots(snap1, snap2)
        d = diff.to_dict()

        assert "env_vars_added" in d
        assert "env_vars_changed" in d
        assert d["env_vars_changed"]["A"]["old"] == "old"
        assert d["env_vars_changed"]["A"]["new"] == "new"
