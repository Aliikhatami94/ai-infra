# ai_infra/llm/tools/custom/mcp_stdio_expose.py
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
        base_dir: Optional[str] = None,
        dry_run: bool = False,
) -> dict:
    """
    Create/update an npx-compatible shim for a Python MCP stdio server.

    Supports sandboxed agents:
      - Pass base_dir to write under a known writable root.
      - Set dry_run=True to return the file contents without touching disk.
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
        base_dir=Path(base_dir) if base_dir else None,
        dry_run=dry_run,
    )

def mcp_expose_remove(
        tool_name: str,
        package_json: str = "package.json",
        bin_dir: str = "src/mcp-shim/bin",
        python_package_root: Optional[str] = None,
        delete_file: bool = False,
        base_dir: Optional[str] = None,
) -> dict:
    """
    Remove the shim mapping from package.json and optionally delete the shim file.

    For sandboxed agents, pass base_dir if package.json/shim live under a writable root.
    """
    return remove_shim(
        tool_name=tool_name,
        package_json=Path(package_json),
        bin_dir=Path(bin_dir),
        python_package_root=python_package_root,
        delete_file=delete_file,
        base_dir=Path(base_dir) if base_dir else None,
    )