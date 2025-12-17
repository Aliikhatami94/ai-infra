"""Project management tools with configurable workspace sandboxing.

Supports two modes:
    1. Repo-sandboxed (default): Auto-detects project root via markers
       (.git, pyproject.toml, package.json, etc.)

    2. Workspace-sandboxed: Explicit directory via set_workspace_root()

Example:
    # Default: repo-sandboxed
    from ai_infra.llm.tools.custom.proj_mgmt import file_read
    content = await file_read("src/main.py")

    # Explicit workspace (e.g., for isolated agent environments)
    from ai_infra.llm.tools.custom.proj_mgmt import set_workspace_root, file_read
    set_workspace_root("/tmp/agent-workspace")
    content = await file_read("input.txt")
"""

from ai_infra.llm.tools.custom.proj_mgmt.main import (
    file_read,
    file_write,
    files_list,
    project_scan,
)
from ai_infra.llm.tools.custom.proj_mgmt.utils import (
    ToolException,
    get_workspace_root,
    set_workspace_root,
)

__all__ = [
    # Tools
    "project_scan",
    "files_list",
    "file_read",
    "file_write",
    # Workspace config
    "set_workspace_root",
    "get_workspace_root",
    # Exceptions
    "ToolException",
]
