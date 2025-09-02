from ai_infra.mcp.server.tools import mcp_from_functions
from ai_infra.llm.tools.custom.proj_mgmt.main import (
    file_read,
    file_write,
    files_list,
    project_scan
)

mcp = mcp_from_functions(
    name="project-management",
    functions=[file_read, file_write, files_list, project_scan]
)

def main():
    mcp.run(transport="stdio")