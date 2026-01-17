"""Shell environment snapshot capture (Phase 16.1).

This module provides functionality to capture the current shell environment
state, including environment variables, aliases, functions, and shell options.

Supports bash, zsh, and fish shells with automatic detection.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ai_infra.logging import get_logger

logger = get_logger("executor.shell")


# =============================================================================
# Constants
# =============================================================================


# Environment variables to exclude from capture (sensitive or system-managed)
EXCLUDED_ENV_VARS = frozenset(
    {
        # Sensitive
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "NPM_TOKEN",
        "PYPI_TOKEN",
        # Session-specific (will vary between sessions)
        "TERM_SESSION_ID",
        "WINDOWID",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "DISPLAY",
        # Process-specific
        "PPID",
        "SHLVL",
        "_",
        "OLDPWD",
    }
)

# Prefixes for env vars to exclude
EXCLUDED_ENV_PREFIXES = (
    "BASH_FUNC_",  # Bash exported functions
    "DIRENV_",  # direnv internal state
    "__",  # Double underscore (usually internal)
)


# =============================================================================
# Shell Types
# =============================================================================


class ShellType(str, Enum):
    """Supported shell types."""

    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"
    SH = "sh"
    UNKNOWN = "unknown"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ShellSnapshot:
    """Captured shell environment state (Phase 16.1.2).

    Attributes:
        env_vars: Environment variables (name -> value).
        aliases: Shell aliases (name -> expansion).
        functions: Shell functions (name -> definition).
        working_dir: Current working directory.
        shell_options: Enabled shell options.
        shell_type: Type of shell (bash, zsh, fish).
        captured_at: Timestamp of capture.
        capture_errors: Any errors during capture.
    """

    env_vars: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    functions: dict[str, str] = field(default_factory=dict)
    working_dir: str = ""
    shell_options: list[str] = field(default_factory=list)
    shell_type: ShellType = ShellType.BASH
    captured_at: datetime = field(default_factory=datetime.now)
    capture_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "env_vars": self.env_vars,
            "aliases": self.aliases,
            "functions": self.functions,
            "working_dir": self.working_dir,
            "shell_options": self.shell_options,
            "shell_type": self.shell_type.value,
            "captured_at": self.captured_at.isoformat(),
            "capture_errors": self.capture_errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShellSnapshot:
        """Create ShellSnapshot from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ShellSnapshot instance.
        """
        return cls(
            env_vars=data.get("env_vars", {}),
            aliases=data.get("aliases", {}),
            functions=data.get("functions", {}),
            working_dir=data.get("working_dir", ""),
            shell_options=data.get("shell_options", []),
            shell_type=ShellType(data.get("shell_type", "bash")),
            captured_at=datetime.fromisoformat(data["captured_at"])
            if "captured_at" in data
            else datetime.now(),
            capture_errors=data.get("capture_errors", []),
        )

    @property
    def env_var_count(self) -> int:
        """Number of captured environment variables."""
        return len(self.env_vars)

    @property
    def alias_count(self) -> int:
        """Number of captured aliases."""
        return len(self.aliases)

    @property
    def function_count(self) -> int:
        """Number of captured functions."""
        return len(self.functions)

    def summary(self) -> str:
        """Get a brief summary of the snapshot.

        Returns:
            Human-readable summary string.
        """
        return (
            f"ShellSnapshot({self.shell_type.value}): "
            f"{self.env_var_count} env vars, "
            f"{self.alias_count} aliases, "
            f"{self.function_count} functions, "
            f"cwd={self.working_dir}"
        )


# =============================================================================
# Shell Detection (Phase 16.1.8)
# =============================================================================


def detect_shell() -> ShellType:
    """Detect the current shell type (Phase 16.1.8).

    Detection order:
    1. $SHELL environment variable
    2. /proc/self/exe (Linux)
    3. ps command output
    4. Default to UNKNOWN

    Returns:
        Detected ShellType.
    """
    # Try $SHELL first
    shell_env = os.environ.get("SHELL", "")
    if shell_env:
        shell_name = os.path.basename(shell_env).lower()
        if "zsh" in shell_name:
            return ShellType.ZSH
        elif "bash" in shell_name:
            return ShellType.BASH
        elif "fish" in shell_name:
            return ShellType.FISH
        elif shell_name == "sh":
            return ShellType.SH

    # Try to find shell executable
    for shell_type, binary in [
        (ShellType.ZSH, "zsh"),
        (ShellType.BASH, "bash"),
        (ShellType.FISH, "fish"),
    ]:
        if shutil.which(binary):
            # Check if it's the current shell via process
            try:
                import subprocess

                result = subprocess.run(
                    ["ps", "-p", str(os.getppid()), "-o", "comm="],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parent_name = result.stdout.strip().lower()
                    if binary in parent_name:
                        return shell_type
            except Exception:
                pass

    logger.debug("Could not detect shell type, defaulting to UNKNOWN")
    return ShellType.UNKNOWN


def _get_shell_binary(shell_type: ShellType) -> str:
    """Get the shell binary path for a shell type.

    Args:
        shell_type: Type of shell.

    Returns:
        Path to shell binary or shell name.
    """
    if shell_type == ShellType.ZSH:
        return shutil.which("zsh") or "zsh"
    elif shell_type == ShellType.FISH:
        return shutil.which("fish") or "fish"
    elif shell_type == ShellType.SH:
        return shutil.which("sh") or "sh"
    else:
        # Default to bash
        return shutil.which("bash") or "bash"


# =============================================================================
# Environment Variable Capture (Phase 16.1.3)
# =============================================================================


async def capture_env_vars(
    exclude_sensitive: bool = True,
    exclude_patterns: list[str] | None = None,
) -> dict[str, str]:
    """Capture environment variables (Phase 16.1.3).

    Args:
        exclude_sensitive: Whether to exclude sensitive env vars.
        exclude_patterns: Additional patterns to exclude (regex).

    Returns:
        Dictionary of environment variable name -> value.
    """
    env_vars: dict[str, str] = {}
    exclude_re = [re.compile(p) for p in (exclude_patterns or [])]

    for name, value in os.environ.items():
        # Skip excluded vars
        if exclude_sensitive and name in EXCLUDED_ENV_VARS:
            continue

        # Skip excluded prefixes
        if any(name.startswith(prefix) for prefix in EXCLUDED_ENV_PREFIXES):
            continue

        # Skip pattern matches
        if any(pattern.match(name) for pattern in exclude_re):
            continue

        env_vars[name] = value

    logger.debug(f"Captured {len(env_vars)} environment variables")
    return env_vars


# =============================================================================
# Alias Capture (Phase 16.1.4)
# =============================================================================


async def capture_aliases(shell_type: ShellType | None = None) -> dict[str, str]:
    """Capture shell aliases (Phase 16.1.4).

    Args:
        shell_type: Shell type (auto-detected if None).

    Returns:
        Dictionary of alias name -> expansion.
    """
    detected_shell = shell_type or detect_shell()
    aliases: dict[str, str] = {}

    if detected_shell == ShellType.FISH:
        # Fish uses `functions` for aliases too
        cmd = ["fish", "-c", "alias"]
    elif detected_shell in (ShellType.BASH, ShellType.ZSH, ShellType.SH):
        # Bash/Zsh use `alias` command
        shell_bin = _get_shell_binary(detected_shell)
        cmd = [shell_bin, "-i", "-c", "alias"]
    else:
        logger.warning(f"Alias capture not supported for {detected_shell}")
        return aliases

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PS1": ""},  # Disable prompt
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        output = stdout.decode("utf-8", errors="replace")

        # Parse alias output
        # Bash/Zsh format: alias name='value' or alias name="value"
        # Fish format: alias name 'value'
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if detected_shell == ShellType.FISH:
                # Fish: alias name 'value' or alias name value
                match = re.match(r"alias\s+(\S+)\s+(.+)", line)
                if match:
                    name, value = match.groups()
                    # Remove quotes if present
                    value = value.strip("'\"")
                    aliases[name] = value
            else:
                # Bash/Zsh: alias name='value' or name="value"
                match = re.match(r"(?:alias\s+)?(\S+)=['\"]?(.+?)['\"]?$", line)
                if match:
                    name, value = match.groups()
                    aliases[name] = value

    except TimeoutError:
        logger.warning("Alias capture timed out")
    except FileNotFoundError:
        logger.warning(f"Shell binary not found for {detected_shell}")
    except Exception as e:
        logger.warning(f"Failed to capture aliases: {e}")

    logger.debug(f"Captured {len(aliases)} aliases")
    return aliases


# =============================================================================
# Function Capture (Phase 16.1.5)
# =============================================================================


async def capture_functions(shell_type: ShellType | None = None) -> dict[str, str]:
    """Capture shell functions (Phase 16.1.5).

    Uses:
    - `declare -f` for bash
    - `functions` or `declare -f` for zsh
    - `functions` for fish

    Args:
        shell_type: Shell type (auto-detected if None).

    Returns:
        Dictionary of function name -> definition.
    """
    detected_shell = shell_type or detect_shell()
    functions: dict[str, str] = {}

    if detected_shell == ShellType.FISH:
        cmd = ["fish", "-c", "functions"]
    elif detected_shell == ShellType.ZSH:
        # Zsh: typeset -f or declare -f
        shell_bin = _get_shell_binary(detected_shell)
        cmd = [shell_bin, "-i", "-c", "typeset -f"]
    elif detected_shell in (ShellType.BASH, ShellType.SH):
        # Bash: declare -f
        shell_bin = _get_shell_binary(detected_shell)
        cmd = [shell_bin, "-i", "-c", "declare -f"]
    else:
        logger.warning(f"Function capture not supported for {detected_shell}")
        return functions

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PS1": ""},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        output = stdout.decode("utf-8", errors="replace")

        if detected_shell == ShellType.FISH:
            # Fish: `functions` lists names, need to get each definition
            func_names = [name.strip() for name in output.split("\n") if name.strip()]
            for name in func_names[:50]:  # Limit to 50 functions
                try:
                    defn_proc = await asyncio.create_subprocess_exec(
                        "fish",
                        "-c",
                        f"functions {name}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    defn_out, _ = await asyncio.wait_for(defn_proc.communicate(), timeout=5.0)
                    functions[name] = defn_out.decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
        else:
            # Bash/Zsh: parse declare -f output
            # Format: func_name () { ... }
            functions = _parse_bash_functions(output)

    except TimeoutError:
        logger.warning("Function capture timed out")
    except FileNotFoundError:
        logger.warning(f"Shell binary not found for {detected_shell}")
    except Exception as e:
        logger.warning(f"Failed to capture functions: {e}")

    logger.debug(f"Captured {len(functions)} functions")
    return functions


def _parse_bash_functions(output: str) -> dict[str, str]:
    """Parse bash/zsh `declare -f` output into functions.

    Args:
        output: Raw output from declare -f.

    Returns:
        Dictionary of function name -> definition.
    """
    functions: dict[str, str] = {}
    current_name: str | None = None
    current_body: list[str] = []
    brace_count = 0

    for line in output.split("\n"):
        # Function definition start: "func_name ()" or "func_name() {"
        if current_name is None:
            match = re.match(r"^(\w+)\s*\(\)\s*$", line.strip())
            if match:
                current_name = match.group(1)
                current_body = [line]
                continue

            # Single line with opening brace
            match = re.match(r"^(\w+)\s*\(\)\s*\{", line.strip())
            if match:
                current_name = match.group(1)
                current_body = [line]
                brace_count = line.count("{") - line.count("}")
                if brace_count == 0:
                    # Single-line function
                    functions[current_name] = line.strip()
                    current_name = None
                    current_body = []
                continue
        else:
            # Inside a function definition
            current_body.append(line)
            brace_count += line.count("{") - line.count("}")

            if brace_count <= 0 and "}" in line:
                # Function definition complete
                functions[current_name] = "\n".join(current_body).strip()
                current_name = None
                current_body = []
                brace_count = 0

    return functions


# =============================================================================
# Shell Options Capture (Phase 16.1.6)
# =============================================================================


async def capture_shell_options(shell_type: ShellType | None = None) -> list[str]:
    """Capture shell options (Phase 16.1.6).

    Uses:
    - `shopt -s` for bash (set options)
    - `setopt` for zsh
    - `status features` for fish

    Args:
        shell_type: Shell type (auto-detected if None).

    Returns:
        List of enabled shell options.
    """
    detected_shell = shell_type or detect_shell()
    options: list[str] = []

    if detected_shell == ShellType.FISH:
        # Fish: status features or set -U
        cmd = ["fish", "-c", "status features"]
    elif detected_shell == ShellType.ZSH:
        # Zsh: setopt (shows set options)
        shell_bin = _get_shell_binary(detected_shell)
        cmd = [shell_bin, "-i", "-c", "setopt"]
    elif detected_shell == ShellType.BASH:
        # Bash: shopt -s (shows enabled options)
        shell_bin = _get_shell_binary(detected_shell)
        cmd = [shell_bin, "-i", "-c", "shopt -s"]
    else:
        logger.warning(f"Shell options capture not supported for {detected_shell}")
        return options

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PS1": ""},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        output = stdout.decode("utf-8", errors="replace")

        if detected_shell == ShellType.BASH:
            # Bash shopt -s output: "option_name    on"
            for line in output.strip().split("\n"):
                line = line.strip()
                if line and "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[1].strip() == "on":
                        options.append(parts[0].strip())
                elif line:
                    # Just the option name
                    options.append(line)
        elif detected_shell == ShellType.ZSH:
            # Zsh setopt output: one option per line
            options = [line.strip() for line in output.split("\n") if line.strip()]
        elif detected_shell == ShellType.FISH:
            # Fish: parse status features output
            for line in output.strip().split("\n"):
                if ":" in line:
                    feature, status = line.split(":", 1)
                    if "enabled" in status.lower() or "on" in status.lower():
                        options.append(feature.strip())

    except TimeoutError:
        logger.warning("Shell options capture timed out")
    except FileNotFoundError:
        logger.warning(f"Shell binary not found for {detected_shell}")
    except Exception as e:
        logger.warning(f"Failed to capture shell options: {e}")

    logger.debug(f"Captured {len(options)} shell options")
    return options


# =============================================================================
# Combined Capture (Phase 16.1.7)
# =============================================================================


async def capture_shell_state(
    shell_type: ShellType | None = None,
    capture_functions_flag: bool = True,
    capture_aliases_flag: bool = True,
    capture_options_flag: bool = True,
    exclude_sensitive_env: bool = True,
) -> ShellSnapshot:
    """Capture complete shell environment state (Phase 16.1.7).

    Combines all capture functions into a single ShellSnapshot.

    Args:
        shell_type: Shell type (auto-detected if None).
        capture_functions_flag: Whether to capture shell functions.
        capture_aliases_flag: Whether to capture aliases.
        capture_options_flag: Whether to capture shell options.
        exclude_sensitive_env: Whether to exclude sensitive env vars.

    Returns:
        Complete ShellSnapshot with all captured state.
    """
    detected_shell = shell_type or detect_shell()
    errors: list[str] = []

    logger.info(f"Capturing shell state for {detected_shell.value}")

    # Capture environment variables (always)
    try:
        env_vars = await capture_env_vars(exclude_sensitive=exclude_sensitive_env)
    except Exception as e:
        env_vars = {}
        errors.append(f"Failed to capture env vars: {e}")

    # Capture aliases
    aliases: dict[str, str] = {}
    if capture_aliases_flag:
        try:
            aliases = await capture_aliases(detected_shell)
        except Exception as e:
            errors.append(f"Failed to capture aliases: {e}")

    # Capture functions
    functions: dict[str, str] = {}
    if capture_functions_flag:
        try:
            functions = await capture_functions(detected_shell)
        except Exception as e:
            errors.append(f"Failed to capture functions: {e}")

    # Capture shell options
    shell_options: list[str] = []
    if capture_options_flag:
        try:
            shell_options = await capture_shell_options(detected_shell)
        except Exception as e:
            errors.append(f"Failed to capture shell options: {e}")

    # Get current working directory
    try:
        working_dir = os.getcwd()
    except OSError as e:
        working_dir = ""
        errors.append(f"Failed to get working directory: {e}")

    snapshot = ShellSnapshot(
        env_vars=env_vars,
        aliases=aliases,
        functions=functions,
        working_dir=working_dir,
        shell_options=shell_options,
        shell_type=detected_shell,
        captured_at=datetime.now(),
        capture_errors=errors,
    )

    logger.info(f"Shell state captured: {snapshot.summary()}")
    if errors:
        logger.warning(f"Capture completed with {len(errors)} error(s)")

    return snapshot
