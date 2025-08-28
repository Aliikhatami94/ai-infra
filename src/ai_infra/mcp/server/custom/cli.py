from ai_infra.mcp.server.tools import mcp_from_tools
from ai_infra.llm.tools.custom.cli import run_command

mcp = mcp_from_tools(
    name="cli",
    tools=[run_command]
)

def main():
    mcp.run(transport="stdio")