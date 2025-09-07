from ai_infra.mcp.server.tools import mcp_from_functions
from ai_infra.llm.tools.custom.cli import run_cli

mcp = mcp_from_functions(
    name="cli",
    functions=[run_cli]
)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()