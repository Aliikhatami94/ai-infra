from ai_infra.mcp.server.tools import tools_from_functions
from ai_infra.llm.tools.custom.proj_mgmt.main import (
    file_read,
    file_write,
    files_list,
    project_scan
)

mcp = tools_from_functions(
    name="project-management",
    tools=[file_read, file_write, files_list, project_scan]
)

def main():
    mcp.run(transport="stdio")