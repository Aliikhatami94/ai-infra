from __future__ import annotations

from ai_infra.mcp.server.tools import mcp_from_functions
from ai_infra.llm.tools.custom.stdio_publisher import mcp_publish_remove, mcp_publish_add, make_executable

mcp = mcp_from_functions(
    name="stdio-exposure-mcp",
    functions=[
        mcp_publish_add,
        mcp_publish_remove,
        make_executable
    ],
)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()