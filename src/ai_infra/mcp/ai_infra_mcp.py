from __future__ import annotations

from enum import Enum

from ai_infra.llm.tools.custom.cli import cli_cmd_help, cli_subcmd_help
from ai_infra.mcp.server.tools import mcp_from_functions

CLI_PROG = "ai-infra"


async def ai_infra_cmd_help() -> str:
    """
    Get help text for ai-infra CLI.
    - Prepares project env without chdir (so we can 'cd' in the command itself).
    - Tries poetry → console script → python -m ai_infra.cli_shim.
    """
    import json

    result = await cli_cmd_help(CLI_PROG)
    return json.dumps(result)


class Subcommand(str, Enum):
    add_publisher = "add-publisher"
    remove_publisher = "remove-publisher"
    chmod_publisher = "chmod"
    chmod_all = "chmod-all"


async def ai_infra_subcmd_help(subcommand: Subcommand) -> str:
    """
    Get help text for a specific subcommand of ai-infra CLI.
    (Enum keeps a tight schema; function signature remains simple.)
    """
    import json

    result = await cli_subcmd_help(CLI_PROG, subcommand)
    return json.dumps(result)


mcp = mcp_from_functions(
    name="ai-infra-cli-mcp",
    functions=[
        ai_infra_cmd_help,
        ai_infra_subcmd_help,
    ],
)

if __name__ == "__main__":
    mcp.run(transport="stdio")
