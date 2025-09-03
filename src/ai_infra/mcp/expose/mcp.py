from __future__ import annotations

from ai_infra.mcp.server.tools import mcp_from_functions
from ai_infra.llm.tools.custom.mcp_stdio_expose import mcp_expose_remove, mcp_expose_add

mcp = mcp_from_functions(
    name="expose-stdio-mcp",
    functions=[mcp_expose_add, mcp_expose_remove],
)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()