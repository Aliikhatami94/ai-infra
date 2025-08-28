from ai_infra.mcp.server.tools import mcp_from_tools
from ai_infra.llm.tools.custom.proj_mgmt.main import (
    file_read,
    file_write,
    files_list,
    project_scan
)

mcp = mcp_from_tools(
    name="project-management",
    tools=[file_read, file_write, files_list, project_scan]
)

def main():
    mcp.run(transport="stdio")