"""CLI MCP Server for shell command execution.

Exposes run_shell functionality via MCP (Model Context Protocol) for
use with Claude Desktop and other MCP clients.

Example:
    # Run as stdio transport (for Claude Desktop)
    python -m ai_infra.mcp.server.custom.cli

    # Or via mcp CLI
    mcp run ai_infra/mcp/server/custom/cli.py
"""

from __future__ import annotations

from typing import Any

from ai_infra.llm.shell.tool import run_shell as _run_shell_tool
from ai_infra.mcp.server.tools import mcp_from_functions


async def run_shell(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute a shell command and return the output.

    Use this tool to run shell commands on the host system.

    Args:
        command: Shell command to execute. Supports pipes, redirects, etc.
        cwd: Working directory for the command. If not specified, uses the
            current working directory.
        timeout: Maximum seconds to wait for the command to complete.
            Default is 120 seconds.

    Returns:
        Dictionary with execution result containing success, exit_code,
        stdout, stderr, command, and timed_out fields.

    Example:
        run_shell("ls -la")
        run_shell("npm install", cwd="/path/to/project")
    """
    return await _run_shell_tool.ainvoke(
        {
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
        }
    )


mcp = mcp_from_functions(name="cli", functions=[run_shell])


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
