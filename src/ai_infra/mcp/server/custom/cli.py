from ai_infra.mcp.server.tools import tools_from_functions
from ai_infra.llm.tools.custom.cli import run_command

mcp = tools_from_functions(
    name="cli",
    tools=[run_command]
)

def main():
    mcp.run(transport="stdio")