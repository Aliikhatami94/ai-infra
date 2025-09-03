from pathlib import Path
from typing import Optional

from ai_infra.mcp.expose.core import add_shim, remove_shim

def mcp_expose_add(
        tool_name: str,
        module: str,
        repo: str,
        ref: str = "main",
        package_json: str = "package.json",
        bin_dir: str = "src/mcp-shim/bin",
        python_package_root: Optional[str] = None,
        package_name: str = "mcp-stdio-expose",
        force: bool = False,
) -> dict:
    """
    Create or update an npx-compatible shim that exposes a Python MCP stdio server.

    What it does:
      - Writes a Node launcher (shim) at <bin_dir>/<tool_name>.js that runs:
          uvx --from git+<repo>@<ref> python -m <module> --transport stdio [args...]
      - Ensures the shim is executable (sets +x).
      - Creates/updates the root package.json and maps:
          "bin": { "<tool_name>": "<bin_dir>/<tool_name>.js" }

    Typical agent use:
      After committing the generated files, clients can launch your MCP server via:
        npx --yes --package=github:<owner>/<repo> <tool_name> [args...]

    Args:
        tool_name: CLI name to expose (e.g. "auth-infra-mcp").
        module: Python module with a `main()` that starts the MCP stdio server
                (e.g. "svc_infra.auth.mcp").
        repo: Git URL of the project for uvx (e.g. "https://github.com/org/repo.git").
        ref: Git ref/branch/tag to pull (default: "main").
        package_json: Path to root package.json to create/update (default: "package.json").
        bin_dir: Directory to write shims into (default: "src/mcp-shim/bin").
        python_package_root: If code lives under src/<pkg>/..., pass "<pkg>" to place
                             shims under src/<pkg>/mcp-shim/bin (optional).
        package_name: Name to use when creating a new package.json (default: "mcp-stdio-expose").
        force: Overwrite an existing shim file if present.

    Returns:
        dict with details of the operation (status, action, paths, etc.).
    """
    return add_shim(
        tool_name=tool_name,
        module=module,
        repo=repo,
        ref=ref,
        package_json=Path(package_json),
        bin_dir=Path(bin_dir),
        python_package_root=python_package_root,
        package_name=package_name,
        force=force,
    )

def mcp_expose_remove(
        tool_name: str,
        package_json: str = "package.json",
        bin_dir: str = "src/mcp-shim/bin",
        python_package_root: Optional[str] = None,
        delete_file: bool = False,
) -> dict:
    """
    Remove an npx shim mapping from package.json and optionally delete the shim file.

    What it does:
      - Removes the "bin" entry "<tool_name>" from package.json (if present).
      - Optionally deletes the local shim file at <bin_dir>/<tool_name>.js.

    Args:
        tool_name: The CLI name previously added to package.json "bin".
        package_json: Path to root package.json (default: "package.json").
        bin_dir: Directory where shim files are stored (default: "src/mcp-shim/bin").
        python_package_root: If shims were placed under src/<pkg>/mcp-shim/bin, pass "<pkg>".
        delete_file: If True, also delete the shim file on disk.

    Returns:
        dict with details of the removal (flags for JSON and file deletion).
    """
    return remove_shim(
        tool_name=tool_name,
        package_json=Path(package_json),
        bin_dir=Path(bin_dir),
        python_package_root=python_package_root,
        delete_file=delete_file,
    )