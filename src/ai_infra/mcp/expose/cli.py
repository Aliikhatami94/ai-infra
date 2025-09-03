from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from .core import add_shim, remove_shim

app = typer.Typer(help="Expose (publish) MCP stdio servers via npx-compatible shims.")

@app.command("add")
def add_cmd(
        tool_name: str = typer.Option(..., help="CLI name (e.g. auth-infra-mcp)"),
        module: str = typer.Option(..., help="Python module (e.g. svc_infra.auth.mcp)"),
        repo: str = typer.Option(..., help="Git repo URL for uvx"),
        ref: str = typer.Option("main", help="Git ref/branch/tag"),
        package_json: Path = typer.Option(Path("package.json")),
        bin_dir: Path = typer.Option(Path("src") / "mcp-shim" / "bin"),
        python_package_root: Optional[str] = typer.Option(None),
        package_name: str = typer.Option("mcp-shims"),
        force: bool = typer.Option(False),
):
    res = add_shim(
        tool_name=tool_name, module=module, repo=repo, ref=ref,
        package_json=package_json, bin_dir=bin_dir,
        python_package_root=python_package_root, package_name=package_name,
        force=force,
    )
    typer.echo(res)

@app.command("remove")
def remove_cmd(
        tool_name: str = typer.Option(...),
        package_json: Path = typer.Option(Path("package.json")),
        bin_dir: Path = typer.Option(Path("src") / "mcp-shim" / "bin"),
        python_package_root: Optional[str] = typer.Option(None),
        delete_file: bool = typer.Option(False),
):
    res = remove_shim(
        tool_name=tool_name, package_json=package_json,
        bin_dir=bin_dir, python_package_root=python_package_root,
        delete_file=delete_file,
    )
    typer.echo(res)

if __name__ == "__main__":
    app()