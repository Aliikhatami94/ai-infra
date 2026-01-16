"""Shell helper functions for common CLI operations.

This module provides convenience functions for common shell operations
that were previously in the old cli.py module.

Phase 1.5 of EXECUTOR_CLI.md - Shell Tool Integration.

Example:
    ```python
    from ai_infra.llm.shell.helpers import cli_help, run_command

    # Get help for a CLI program
    help_info = await cli_help("poetry")

    # Run a command with structured result
    result = await run_command("ls -la")
    if result["success"]:
        print(result["stdout"])
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_infra.llm.shell.tool import run_shell
from ai_infra.llm.shell.types import HostExecutionPolicy, ShellConfig, ShellResult

__all__ = [
    "cli_help",
    "cli_cmd_help",
    "cli_subcmd_help",
    "check_command_exists",
    "run_command",
    "run_command_sync",
]


# =============================================================================
# CLI Help Functions
# =============================================================================


async def cli_cmd_help(program: str) -> dict[str, str | bool]:
    """Get help text for a CLI program.

    This is a compatibility wrapper that matches the old cli.py interface.
    Use cli_help() for new code.

    Args:
        program: CLI program name (e.g., "poetry", "npm", "git")

    Returns:
        Dictionary with:
        - ok: Whether the command succeeded
        - help: The help text from stdout
        - error: Error message if failed (empty string on success)

    Example:
        >>> result = await cli_cmd_help("poetry")
        >>> if result["ok"]:
        ...     print(result["help"])
    """
    result = await run_shell.ainvoke({"command": f"{program} --help"})
    return {
        "ok": result["success"],
        "help": result["stdout"],
        "error": result["stderr"] if not result["success"] else "",
    }


async def cli_subcmd_help(program: str, subcommand: str) -> dict[str, str | bool]:
    """Get help text for a CLI program subcommand.

    This is a compatibility wrapper that matches the old cli.py interface.
    Use cli_help() for new code.

    Args:
        program: CLI program name (e.g., "poetry", "npm", "git")
        subcommand: Subcommand to get help for (e.g., "add", "install")

    Returns:
        Dictionary with:
        - ok: Whether the command succeeded
        - help: The help text from stdout
        - error: Error message if failed (empty string on success)

    Example:
        >>> result = await cli_subcmd_help("poetry", "add")
        >>> if result["ok"]:
        ...     print(result["help"])
    """
    # Handle enum values (e.g., from Subcommand enum in ai_infra_mcp.py)
    if hasattr(subcommand, "value"):
        subcommand = subcommand.value

    result = await run_shell.ainvoke({"command": f"{program} {subcommand} --help"})
    return {
        "ok": result["success"],
        "help": result["stdout"],
        "error": result["stderr"] if not result["success"] else "",
    }


async def check_command_exists(command: str) -> bool:
    """Check if a command is available on the system.

    Args:
        command: Command name to check (e.g., "poetry", "npm")

    Returns:
        True if the command exists and is executable, False otherwise.

    Example:
        >>> if await check_command_exists("poetry"):
        ...     print("Poetry is available")
    """
    result = await run_shell.ainvoke({"command": f"which {command}"})
    return result["success"]


async def cli_help(
    program: str,
    subcommand: str | None = None,
    *,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Get CLI help text for a program or subcommand.

    Args:
        program: CLI program name (e.g., "poetry", "npm", "git")
        subcommand: Optional subcommand (e.g., "install", "add")
        cwd: Working directory for the command

    Returns:
        Dictionary with:
        - ok: Whether the command succeeded
        - help: The help text from stdout
        - error: Error message if failed (None on success)
        - program: The program name
        - subcommand: The subcommand (if provided)

    Example:
        >>> result = await cli_help("poetry")
        >>> print(result["help"])
        Poetry (version 1.8.0)
        ...

        >>> result = await cli_help("poetry", "add")
        >>> print(result["help"])
        Add a new dependency to pyproject.toml
        ...
    """
    if subcommand:
        cmd = f"{program} {subcommand} --help"
    else:
        cmd = f"{program} --help"

    result = await run_shell.ainvoke(
        {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
        }
    )

    return {
        "ok": result["success"],
        "help": result["stdout"],
        "error": result["stderr"] if not result["success"] else None,
        "program": program,
        "subcommand": subcommand,
    }


# =============================================================================
# Command Execution Functions
# =============================================================================


async def run_command(
    command: str,
    *,
    cwd: str | Path | None = None,
    timeout: float = 120.0,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Run a shell command and return structured result.

    This is a convenience wrapper around run_shell that provides
    a simpler interface for common use cases.

    Args:
        command: Shell command to execute
        cwd: Working directory for the command
        timeout: Timeout in seconds (default: 120)
        raise_on_error: Whether to raise RuntimeError on failure

    Returns:
        Dictionary with:
        - success: Whether the command succeeded
        - exit_code: Exit code of the command
        - stdout: Standard output
        - stderr: Standard error
        - command: The command that was executed

    Raises:
        RuntimeError: If raise_on_error=True and command fails

    Example:
        >>> result = await run_command("ls -la")
        >>> if result["success"]:
        ...     print(result["stdout"])

        >>> # With error raising
        >>> result = await run_command("make build", raise_on_error=True)
    """
    result = await run_shell.ainvoke(
        {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "timeout": int(timeout),
        }
    )

    if raise_on_error and not result["success"]:
        raise RuntimeError(
            f"Command failed with code {result['exit_code']}\n"
            f"STDOUT:\n{result['stdout']}\n"
            f"STDERR:\n{result['stderr']}"
        )

    return result


def run_command_sync(
    command: str,
    *,
    cwd: str | Path | None = None,
    timeout: float = 120.0,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Synchronous version of run_command.

    Uses HostExecutionPolicy directly for synchronous contexts.

    Args:
        command: Shell command to execute
        cwd: Working directory for the command
        timeout: Timeout in seconds (default: 120)
        raise_on_error: Whether to raise RuntimeError on failure

    Returns:
        Dictionary with success, exit_code, stdout, stderr, command

    Raises:
        RuntimeError: If raise_on_error=True and command fails
    """
    import asyncio

    async def _run() -> ShellResult:
        policy = HostExecutionPolicy()
        config = ShellConfig(
            timeout=timeout,
            cwd=Path(cwd) if cwd else None,
        )
        return await policy.execute(command, config)

    # Run in event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # We're in an async context, create a new loop in a thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _run())
            shell_result = future.result()
    else:
        shell_result = asyncio.run(_run())

    result = shell_result.to_dict()

    if raise_on_error and not result["success"]:
        raise RuntimeError(
            f"Command failed with code {result['exit_code']}\n"
            f"STDOUT:\n{result['stdout']}\n"
            f"STDERR:\n{result['stderr']}"
        )

    return result
