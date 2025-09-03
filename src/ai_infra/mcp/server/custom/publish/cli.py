from __future__ import annotations
from pathlib import Path
from typing import Optional
import json, typer

from .core import add_shim, remove_shim, _ensure_executable

app = typer.Typer(help="publish (publish) MCP stdio servers via npx-compatible shims.")

def _echo(obj, as_json: bool):
    if as_json:
        typer.echo(json.dumps(obj, indent=2))
    else:
        typer.echo(obj)

@app.command("add")
def add_cmd(
        tool_name: str = typer.Option(..., help="CLI name to publish (e.g. auth-infra-mcp)"),
        module: str = typer.Option(..., help="Python module to run (e.g. svc_infra.auth.mcp)"),
        repo: str = typer.Option(..., help="Git repo URL or GitHub shorthand (owner/repo, github:owner/repo, SSH, or HTTPS)"),
        ref: str = typer.Option("main", help="Git ref/branch/tag (default: main)"),
        package_json: Path = typer.Option(Path("package.json"), help="Path to root package.json"),
        bin_dir: Path = typer.Option(Path("mcp-shim") / "bin", help="Where to write shims (default: mcp-shim/bin)"),
        package_name: str = typer.Option("mcp-stdio-publish", help="package.json:name if creating a new file"),
        force: bool = typer.Option(False, help="Overwrite existing shim file if present"),
        base_dir: Optional[Path] = typer.Option(None, help="Write under this repo root (useful in sandboxed runners)"),
        dry_run: bool = typer.Option(False, help="Emit file contents without touching disk"),
        json_out: bool = typer.Option(True, help="Print machine-readable JSON (default: true)"),
):
    """
    Create/Update:
      - <base_dir>/package.json (adds/updates 'bin' entry)
      - <base_dir>/<bin_dir>/<tool_name>.js (uvx + python -m shim)
    """
    res = add_shim(
        tool_name=tool_name,
        module=module,
        repo=repo,
        ref=ref,
        package_json=package_json,
        bin_dir=bin_dir,
        package_name=package_name,
        force=force,
        base_dir=base_dir,
        dry_run=dry_run,
    )
    _echo(res, as_json=json_out)

@app.command("remove")
def remove_cmd(
        tool_name: str = typer.Option(..., help="CLI name to remove"),
        package_json: Path = typer.Option(Path("package.json")),
        bin_dir: Path = typer.Option(Path("mcp-shim") / "bin"),
        delete_file: bool = typer.Option(False, help="Also delete the shim file"),
        base_dir: Optional[Path] = typer.Option(None),
        json_out: bool = typer.Option(True),
):
    """
    Remove the bin mapping from package.json and optionally delete the shim file.
    """
    res = remove_shim(
        tool_name=tool_name,
        package_json=package_json,
        bin_dir=bin_dir,
        delete_file=delete_file,
        base_dir=base_dir,
    )
    _echo(res, as_json=json_out)

@app.command("chmod")
def chmod_cmd(
        path: Path = typer.Argument(..., help="Shim file path (e.g. mcp-shim/bin/auth-infra-mcp.js)"),
        json_out: bool = typer.Option(True),
):
    """
    Mark a single shim as executable (chmod +x).
    """
    _ensure_executable(path)  # or call your public make_executable([path])
    res = {"status": "ok", "path": str(path.resolve()), "action": "executable_set"}
    _echo(res, as_json=json_out)

@app.command("chmod-all")
def chmod_all_cmd(
        bin_dir: Path = typer.Option(Path("mcp-shim") / "bin", help="Directory containing shim .js files"),
        json_out: bool = typer.Option(True),
):
    """
    Mark all .js files in <bin_dir> as executable.
    """
    bin_dir = bin_dir.resolve()
    updated = []
    if bin_dir.exists():
        for p in bin_dir.glob("*.js"):
            _ensure_executable(p)
            updated.append(str(p))
    res = {"status": "ok", "bin_dir": str(bin_dir), "updated": updated}
    _echo(res, as_json=json_out)

if __name__ == "__main__":
    app()