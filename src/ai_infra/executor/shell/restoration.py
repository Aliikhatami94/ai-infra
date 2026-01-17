"""Shell environment restoration (Phase 16.3).

This module provides functionality to restore shell environment state
from a ShellSnapshot, including environment variables, aliases, functions,
and working directory.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_infra.logging import get_logger

from .snapshot import ShellSnapshot, ShellType

logger = get_logger("executor.shell")


# =============================================================================
# Constants
# =============================================================================


# Environment variables that should never be modified (Phase 16.3.7)
PROTECTED_ENV_VARS = frozenset(
    {
        # Critical system variables
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        # Process identity
        "UID",
        "EUID",
        "GID",
        "EGID",
        "PPID",
        "PID",
        # Security-sensitive
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    }
)

# Environment variables that require extra caution
CAUTION_ENV_VARS = frozenset(
    {
        "PYTHONPATH",
        "NODE_PATH",
        "GOPATH",
        "JAVA_HOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    }
)


# =============================================================================
# Restoration Result
# =============================================================================


@dataclass
class RestorationResult:
    """Result of a restoration operation.

    Attributes:
        success: Whether restoration completed without errors.
        env_vars_restored: Number of env vars restored.
        env_vars_skipped: Number of env vars skipped (protected/caution).
        aliases_restored: Number of aliases included in script.
        functions_restored: Number of functions included in script.
        working_dir_restored: Whether working directory was changed.
        errors: List of errors encountered.
        warnings: List of warnings (e.g., skipped protected vars).
        script_generated: Whether a restoration script was generated.
        script_path: Path to generated script (if any).
    """

    success: bool = True
    env_vars_restored: int = 0
    env_vars_skipped: int = 0
    aliases_restored: int = 0
    functions_restored: int = 0
    working_dir_restored: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    script_generated: bool = False
    script_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "env_vars_restored": self.env_vars_restored,
            "env_vars_skipped": self.env_vars_skipped,
            "aliases_restored": self.aliases_restored,
            "functions_restored": self.functions_restored,
            "working_dir_restored": self.working_dir_restored,
            "errors": self.errors,
            "warnings": self.warnings,
            "script_generated": self.script_generated,
            "script_path": self.script_path,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        parts = []
        if self.env_vars_restored:
            parts.append(f"{self.env_vars_restored} env vars")
        if self.aliases_restored:
            parts.append(f"{self.aliases_restored} aliases")
        if self.functions_restored:
            parts.append(f"{self.functions_restored} functions")
        if self.working_dir_restored:
            parts.append("cwd")

        status = "OK" if self.success else "FAILED"
        restored = ", ".join(parts) if parts else "nothing"
        return f"Restoration [{status}]: {restored}"


# =============================================================================
# Safety Checks (Phase 16.3.7)
# =============================================================================


def is_protected_env_var(name: str) -> bool:
    """Check if an environment variable is protected (Phase 16.3.7).

    Protected variables should never be modified as they could
    break the system or pose security risks.

    Args:
        name: Environment variable name.

    Returns:
        True if the variable is protected.
    """
    return name in PROTECTED_ENV_VARS


def is_caution_env_var(name: str) -> bool:
    """Check if an environment variable requires caution.

    Caution variables can be modified but should be logged.

    Args:
        name: Environment variable name.

    Returns:
        True if the variable requires caution.
    """
    return name in CAUTION_ENV_VARS


def validate_env_var_value(name: str, value: str) -> tuple[bool, str | None]:
    """Validate an environment variable value for safety.

    Args:
        name: Variable name.
        value: Variable value.

    Returns:
        Tuple of (is_safe, warning_message).
    """
    # Check for shell injection patterns
    dangerous_patterns = [
        "$(",
        "`",  # Command substitution
        "&&",
        "||",
        ";",  # Command chaining
        "|",  # Pipes
        ">",
        "<",  # Redirections
    ]

    for pattern in dangerous_patterns:
        if pattern in value:
            return False, f"Potentially dangerous pattern '{pattern}' in {name}"

    # Check for excessively long values
    if len(value) > 10000:
        return False, f"Value for {name} exceeds safe length ({len(value)} chars)"

    return True, None


# =============================================================================
# Restore Environment Variables (Phase 16.3.1)
# =============================================================================


def restore_env_vars(
    snapshot: ShellSnapshot,
    skip_protected: bool = True,
    skip_caution: bool = False,
    validate_values: bool = True,
    dry_run: bool = False,
) -> RestorationResult:
    """Restore environment variables from snapshot (Phase 16.3.1).

    Args:
        snapshot: ShellSnapshot containing env vars to restore.
        skip_protected: Skip protected variables (recommended).
        skip_caution: Skip caution variables.
        validate_values: Validate values for safety.
        dry_run: If True, don't actually set variables.

    Returns:
        RestorationResult with counts and any errors.
    """
    result = RestorationResult()

    for name, value in snapshot.env_vars.items():
        # Check protected
        if skip_protected and is_protected_env_var(name):
            result.env_vars_skipped += 1
            result.warnings.append(f"Skipped protected variable: {name}")
            continue

        # Check caution
        if skip_caution and is_caution_env_var(name):
            result.env_vars_skipped += 1
            result.warnings.append(f"Skipped caution variable: {name}")
            continue

        # Validate value
        if validate_values:
            is_safe, warning = validate_env_var_value(name, value)
            if not is_safe:
                result.env_vars_skipped += 1
                result.warnings.append(warning or f"Unsafe value for {name}")
                continue

        # Set the variable
        if not dry_run:
            try:
                os.environ[name] = value
                result.env_vars_restored += 1

                if is_caution_env_var(name):
                    logger.warning(f"Restored caution variable: {name}")
            except Exception as e:
                result.errors.append(f"Failed to set {name}: {e}")
                result.success = False
        else:
            result.env_vars_restored += 1

    logger.info(f"Restored {result.env_vars_restored} env vars (skipped {result.env_vars_skipped})")
    return result


# =============================================================================
# Restore Working Directory (Phase 16.3.4)
# =============================================================================


def restore_working_dir(
    snapshot: ShellSnapshot,
    create_if_missing: bool = False,
    dry_run: bool = False,
) -> RestorationResult:
    """Restore working directory from snapshot (Phase 16.3.4).

    Args:
        snapshot: ShellSnapshot containing working directory.
        create_if_missing: Create directory if it doesn't exist.
        dry_run: If True, don't actually change directory.

    Returns:
        RestorationResult with status.
    """
    result = RestorationResult()

    if not snapshot.working_dir:
        result.warnings.append("No working directory in snapshot")
        return result

    target_dir = Path(snapshot.working_dir)

    # Check if directory exists
    if not target_dir.exists():
        if create_if_missing:
            if not dry_run:
                try:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created missing directory: {target_dir}")
                except Exception as e:
                    result.errors.append(f"Failed to create directory: {e}")
                    result.success = False
                    return result
        else:
            result.errors.append(f"Directory does not exist: {target_dir}")
            result.success = False
            return result

    # Change directory
    if not dry_run:
        try:
            os.chdir(target_dir)
            result.working_dir_restored = True
            logger.info(f"Changed working directory to: {target_dir}")
        except PermissionError:
            result.errors.append(f"Permission denied: {target_dir}")
            result.success = False
        except Exception as e:
            result.errors.append(f"Failed to change directory: {e}")
            result.success = False
    else:
        result.working_dir_restored = True

    return result


# =============================================================================
# Generate Restoration Script (Phase 16.3.6)
# =============================================================================


def _escape_for_shell(value: str, shell_type: ShellType) -> str:
    """Escape a value for safe inclusion in shell script.

    Args:
        value: Value to escape.
        shell_type: Target shell type.

    Returns:
        Escaped value.
    """
    if shell_type == ShellType.FISH:
        # Fish uses different escaping
        return value.replace("\\", "\\\\").replace("'", "\\'")
    else:
        # Bash/Zsh: use shlex for proper escaping
        return shlex.quote(value)


def generate_restore_script(
    snapshot: ShellSnapshot,
    output_path: str | Path | None = None,
    include_env_vars: bool = True,
    include_aliases: bool = True,
    include_functions: bool = True,
    include_working_dir: bool = True,
    skip_protected: bool = True,
) -> tuple[str, RestorationResult]:
    """Generate a shell script to restore environment (Phase 16.3.6).

    This creates a script that can be sourced to restore the shell
    environment manually.

    Args:
        snapshot: ShellSnapshot to restore from.
        output_path: Path to write script (optional).
        include_env_vars: Include environment variables.
        include_aliases: Include aliases.
        include_functions: Include functions.
        include_working_dir: Include cd to working directory.
        skip_protected: Skip protected env vars.

    Returns:
        Tuple of (script_content, RestorationResult).
    """
    result = RestorationResult()
    shell_type = snapshot.shell_type
    lines: list[str] = []

    # Script header
    if shell_type == ShellType.FISH:
        lines.append("#!/usr/bin/env fish")
    else:
        lines.append("#!/usr/bin/env bash")

    lines.append("# Shell environment restoration script")
    lines.append(f"# Generated from snapshot captured at {snapshot.captured_at}")
    lines.append(f"# Shell type: {shell_type.value}")
    lines.append("")

    # Environment variables (Phase 16.3.1 via script)
    if include_env_vars and snapshot.env_vars:
        lines.append("# Environment Variables")
        for name, value in sorted(snapshot.env_vars.items()):
            if skip_protected and is_protected_env_var(name):
                result.env_vars_skipped += 1
                lines.append(f"# Skipped protected: {name}")
                continue

            # Validate
            is_safe, warning = validate_env_var_value(name, value)
            if not is_safe:
                result.env_vars_skipped += 1
                lines.append(f"# Skipped unsafe: {name} ({warning})")
                continue

            escaped_value = _escape_for_shell(value, shell_type)
            if shell_type == ShellType.FISH:
                lines.append(f"set -gx {name} {escaped_value}")
            else:
                lines.append(f"export {name}={escaped_value}")
            result.env_vars_restored += 1

        lines.append("")

    # Aliases (Phase 16.3.2 via script)
    if include_aliases and snapshot.aliases:
        lines.append("# Aliases")
        for name, expansion in sorted(snapshot.aliases.items()):
            escaped = _escape_for_shell(expansion, shell_type)
            if shell_type == ShellType.FISH:
                lines.append(f"alias {name} {escaped}")
            else:
                lines.append(f"alias {name}={escaped}")
            result.aliases_restored += 1

        lines.append("")

    # Functions (Phase 16.3.3 via script)
    if include_functions and snapshot.functions:
        lines.append("# Functions")
        for name, definition in sorted(snapshot.functions.items()):
            # Functions are already in shell syntax
            lines.append(definition)
            lines.append("")
            result.functions_restored += 1

    # Working directory (Phase 16.3.4 via script)
    if include_working_dir and snapshot.working_dir:
        lines.append("# Working Directory")
        escaped_dir = _escape_for_shell(snapshot.working_dir, shell_type)
        lines.append(
            f"cd {escaped_dir} 2>/dev/null || echo 'Warning: Could not cd to {snapshot.working_dir}'"
        )
        result.working_dir_restored = True
        lines.append("")

    # Footer
    lines.append("# End of restoration script")

    script_content = "\n".join(lines)

    # Write to file if path provided
    if output_path:
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script_content, encoding="utf-8")
            # Make executable
            path.chmod(0o755)
            result.script_generated = True
            result.script_path = str(path)
            logger.info(f"Generated restoration script: {path}")
        except Exception as e:
            result.errors.append(f"Failed to write script: {e}")
            result.success = False

    return script_content, result


# =============================================================================
# Restore Aliases (Phase 16.3.2)
# =============================================================================


def restore_aliases(
    snapshot: ShellSnapshot,
    dry_run: bool = False,
) -> tuple[str, RestorationResult]:
    """Generate script to restore aliases (Phase 16.3.2).

    Note: Aliases cannot be directly set from Python. This generates
    a script that can be sourced to define aliases.

    Args:
        snapshot: ShellSnapshot containing aliases.
        dry_run: If True, just return the script.

    Returns:
        Tuple of (alias_script, RestorationResult).
    """
    result = RestorationResult()
    shell_type = snapshot.shell_type
    lines: list[str] = []

    for name, expansion in snapshot.aliases.items():
        escaped = _escape_for_shell(expansion, shell_type)
        if shell_type == ShellType.FISH:
            lines.append(f"alias {name} {escaped}")
        else:
            lines.append(f"alias {name}={escaped}")
        result.aliases_restored += 1

    script = "\n".join(lines)
    logger.debug(f"Generated alias script with {result.aliases_restored} aliases")
    return script, result


# =============================================================================
# Restore Functions (Phase 16.3.3)
# =============================================================================


def restore_functions(
    snapshot: ShellSnapshot,
    dry_run: bool = False,
) -> tuple[str, RestorationResult]:
    """Generate script to restore functions (Phase 16.3.3).

    Note: Functions cannot be directly set from Python. This generates
    a script that can be sourced to define functions.

    Args:
        snapshot: ShellSnapshot containing functions.
        dry_run: If True, just return the script.

    Returns:
        Tuple of (function_script, RestorationResult).
    """
    result = RestorationResult()
    lines: list[str] = []

    for name, definition in snapshot.functions.items():
        lines.append(definition)
        lines.append("")
        result.functions_restored += 1

    script = "\n".join(lines)
    logger.debug(f"Generated function script with {result.functions_restored} functions")
    return script, result


# =============================================================================
# Combined Restoration (Phase 16.3.5)
# =============================================================================


def restore_shell_state(
    snapshot: ShellSnapshot,
    restore_env: bool = True,
    restore_cwd: bool = True,
    generate_script: bool = True,
    script_path: str | Path | None = None,
    skip_protected: bool = True,
    dry_run: bool = False,
) -> RestorationResult:
    """Restore complete shell state from snapshot (Phase 16.3.5).

    This combines all restoration operations:
    - Environment variables (directly in Python)
    - Working directory (directly in Python)
    - Aliases and functions (via generated script)

    Args:
        snapshot: ShellSnapshot to restore from.
        restore_env: Restore environment variables.
        restore_cwd: Restore working directory.
        generate_script: Generate script for aliases/functions.
        script_path: Path for generated script.
        skip_protected: Skip protected env vars.
        dry_run: If True, don't make actual changes.

    Returns:
        Combined RestorationResult.
    """
    result = RestorationResult()

    logger.info(f"Restoring shell state from snapshot: {snapshot.summary()}")

    # Restore environment variables (Phase 16.3.1)
    if restore_env:
        env_result = restore_env_vars(
            snapshot,
            skip_protected=skip_protected,
            dry_run=dry_run,
        )
        result.env_vars_restored = env_result.env_vars_restored
        result.env_vars_skipped = env_result.env_vars_skipped
        result.errors.extend(env_result.errors)
        result.warnings.extend(env_result.warnings)
        if not env_result.success:
            result.success = False

    # Restore working directory (Phase 16.3.4)
    if restore_cwd:
        cwd_result = restore_working_dir(
            snapshot,
            dry_run=dry_run,
        )
        result.working_dir_restored = cwd_result.working_dir_restored
        result.errors.extend(cwd_result.errors)
        result.warnings.extend(cwd_result.warnings)
        if not cwd_result.success:
            result.success = False

    # Generate script for aliases and functions (Phase 16.3.2, 16.3.3, 16.3.6)
    if generate_script and (snapshot.aliases or snapshot.functions):
        script_content, script_result = generate_restore_script(
            snapshot,
            output_path=script_path,
            include_env_vars=False,  # Already restored directly
            include_aliases=True,
            include_functions=True,
            include_working_dir=False,  # Already restored directly
            skip_protected=skip_protected,
        )
        result.aliases_restored = script_result.aliases_restored
        result.functions_restored = script_result.functions_restored
        result.script_generated = script_result.script_generated
        result.script_path = script_result.script_path
        result.errors.extend(script_result.errors)
        if not script_result.success:
            result.success = False

        if script_content and not script_path:
            # Log instructions for manual sourcing
            logger.info(
                "To restore aliases and functions, source the following:\n"
                'eval "$(<generated_script>)"\n'
                "Or run: source <script_path>"
            )

    logger.info(f"Shell state restoration complete: {result.summary()}")
    return result


# =============================================================================
# Diff Utilities (for Phase 16.4.5)
# =============================================================================


@dataclass
class SnapshotDiff:
    """Difference between two snapshots.

    Attributes:
        env_vars_added: New env vars in snapshot2.
        env_vars_removed: Env vars only in snapshot1.
        env_vars_changed: Env vars with different values.
        aliases_added: New aliases in snapshot2.
        aliases_removed: Aliases only in snapshot1.
        aliases_changed: Aliases with different values.
        functions_added: New functions in snapshot2.
        functions_removed: Functions only in snapshot1.
        functions_changed: Functions with different definitions.
        working_dir_changed: Whether working directory changed.
        old_working_dir: Previous working directory.
        new_working_dir: New working directory.
    """

    env_vars_added: dict[str, str] = field(default_factory=dict)
    env_vars_removed: dict[str, str] = field(default_factory=dict)
    env_vars_changed: dict[str, tuple[str, str]] = field(default_factory=dict)
    aliases_added: dict[str, str] = field(default_factory=dict)
    aliases_removed: dict[str, str] = field(default_factory=dict)
    aliases_changed: dict[str, tuple[str, str]] = field(default_factory=dict)
    functions_added: dict[str, str] = field(default_factory=dict)
    functions_removed: dict[str, str] = field(default_factory=dict)
    functions_changed: dict[str, tuple[str, str]] = field(default_factory=dict)
    working_dir_changed: bool = False
    old_working_dir: str = ""
    new_working_dir: str = ""

    @property
    def has_changes(self) -> bool:
        """Check if there are any differences."""
        return (
            bool(self.env_vars_added)
            or bool(self.env_vars_removed)
            or bool(self.env_vars_changed)
            or bool(self.aliases_added)
            or bool(self.aliases_removed)
            or bool(self.aliases_changed)
            or bool(self.functions_added)
            or bool(self.functions_removed)
            or bool(self.functions_changed)
            or self.working_dir_changed
        )

    def summary(self) -> str:
        """Get human-readable summary of changes."""
        parts = []

        env_changes = (
            len(self.env_vars_added) + len(self.env_vars_removed) + len(self.env_vars_changed)
        )
        if env_changes:
            parts.append(
                f"env: +{len(self.env_vars_added)} "
                f"-{len(self.env_vars_removed)} "
                f"~{len(self.env_vars_changed)}"
            )

        alias_changes = (
            len(self.aliases_added) + len(self.aliases_removed) + len(self.aliases_changed)
        )
        if alias_changes:
            parts.append(
                f"aliases: +{len(self.aliases_added)} "
                f"-{len(self.aliases_removed)} "
                f"~{len(self.aliases_changed)}"
            )

        func_changes = (
            len(self.functions_added) + len(self.functions_removed) + len(self.functions_changed)
        )
        if func_changes:
            parts.append(
                f"functions: +{len(self.functions_added)} "
                f"-{len(self.functions_removed)} "
                f"~{len(self.functions_changed)}"
            )

        if self.working_dir_changed:
            parts.append(f"cwd: {self.old_working_dir} → {self.new_working_dir}")

        if not parts:
            return "No changes"

        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "env_vars_added": self.env_vars_added,
            "env_vars_removed": self.env_vars_removed,
            "env_vars_changed": {
                k: {"old": v[0], "new": v[1]} for k, v in self.env_vars_changed.items()
            },
            "aliases_added": self.aliases_added,
            "aliases_removed": self.aliases_removed,
            "aliases_changed": {
                k: {"old": v[0], "new": v[1]} for k, v in self.aliases_changed.items()
            },
            "functions_added": list(self.functions_added.keys()),
            "functions_removed": list(self.functions_removed.keys()),
            "functions_changed": list(self.functions_changed.keys()),
            "working_dir_changed": self.working_dir_changed,
            "old_working_dir": self.old_working_dir,
            "new_working_dir": self.new_working_dir,
        }


def diff_snapshots(
    snapshot1: ShellSnapshot,
    snapshot2: ShellSnapshot,
) -> SnapshotDiff:
    """Compare two snapshots and return differences.

    Args:
        snapshot1: First (older) snapshot.
        snapshot2: Second (newer) snapshot.

    Returns:
        SnapshotDiff with all differences.
    """
    diff = SnapshotDiff()

    # Compare env vars
    keys1 = set(snapshot1.env_vars.keys())
    keys2 = set(snapshot2.env_vars.keys())

    for key in keys2 - keys1:
        diff.env_vars_added[key] = snapshot2.env_vars[key]

    for key in keys1 - keys2:
        diff.env_vars_removed[key] = snapshot1.env_vars[key]

    for key in keys1 & keys2:
        if snapshot1.env_vars[key] != snapshot2.env_vars[key]:
            diff.env_vars_changed[key] = (
                snapshot1.env_vars[key],
                snapshot2.env_vars[key],
            )

    # Compare aliases
    alias_keys1 = set(snapshot1.aliases.keys())
    alias_keys2 = set(snapshot2.aliases.keys())

    for key in alias_keys2 - alias_keys1:
        diff.aliases_added[key] = snapshot2.aliases[key]

    for key in alias_keys1 - alias_keys2:
        diff.aliases_removed[key] = snapshot1.aliases[key]

    for key in alias_keys1 & alias_keys2:
        if snapshot1.aliases[key] != snapshot2.aliases[key]:
            diff.aliases_changed[key] = (
                snapshot1.aliases[key],
                snapshot2.aliases[key],
            )

    # Compare functions
    func_keys1 = set(snapshot1.functions.keys())
    func_keys2 = set(snapshot2.functions.keys())

    for key in func_keys2 - func_keys1:
        diff.functions_added[key] = snapshot2.functions[key]

    for key in func_keys1 - func_keys2:
        diff.functions_removed[key] = snapshot1.functions[key]

    for key in func_keys1 & func_keys2:
        if snapshot1.functions[key] != snapshot2.functions[key]:
            diff.functions_changed[key] = (
                snapshot1.functions[key],
                snapshot2.functions[key],
            )

    # Compare working directory
    if snapshot1.working_dir != snapshot2.working_dir:
        diff.working_dir_changed = True
        diff.old_working_dir = snapshot1.working_dir
        diff.new_working_dir = snapshot2.working_dir

    return diff
