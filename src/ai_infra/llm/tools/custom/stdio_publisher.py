from __future__ import annotations
import re, stat
from pathlib import Path, PurePosixPath
from typing import Optional, Union, Iterable

from ai_infra.mcp.server.custom.publish.core import add_shim, remove_shim

_GH_SSH   = re.compile(r"^git@github\.com:(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+)(?:\.git)?$")
_GH_HTTPS = re.compile(r"^https?://github\.com/(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+)(?:\.git)?$")
_GH_SHORT = re.compile(r"^(?:github:)?(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+)$")

def _ensure_git_suffix(url: str) -> str:
    return url if url.endswith(".git") else f"{url}.git"

def _normalize_repo(repo: str) -> str:
    repo = repo.strip()
    m = _GH_HTTPS.match(repo)
    if m:
        return _ensure_git_suffix(f"https://github.com/{m.group('owner')}/{m.group('repo')}")
    m = _GH_SSH.match(repo)
    if m:
        return _ensure_git_suffix(f"https://github.com/{m.group('owner')}/{m.group('repo')}")
    m = _GH_SHORT.match(repo)
    if m:
        return _ensure_git_suffix(f"https://github.com/{m.group('owner')}/{m.group('repo')}")
    return repo  # non-GitHub or already normalized

def _infer_pkg_root(module: str) -> Optional[str]:
    head = (module or "").split(".", 1)[0].strip()
    return head or None

def _clean_pkg_root(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = str(PurePosixPath(value)).lstrip("/")
    if cleaned.startswith("src/"):
        cleaned = cleaned[4:]
    # keep only first segment (package name)
    return (cleaned.split("/", 1)[0] or None)

def _coerce_bin_dir(bin_dir: Optional[str]) -> Path:
    if bin_dir:
        return Path(bin_dir)
    return Path("mcp-shim") / "bin"

def mcp_publish_add(
        tool_name: str,
        module: str,
        repo: str,
        ref: str = "main",
        package_json: str = "package.json",
        bin_dir: str = "mcp-shim/bin",
        package_name: str = "mcp-stdio-publish",
        force: bool = False,
        base_dir: Optional[str] = None,
        dry_run: bool = False,
) -> dict:
    """
    Create or update an executable shim that publishs a Python MCP stdio server via Node's "bin" interface.

    What it delivers
    - A Node.js shim file at <bin_dir>/<tool_name>.js that runs: uvx --from git+<repo>@<ref> python -m <module> --transport stdio <args>.
    - A package.json "bin" mapping of <tool_name> -> relative path to that shim, so it can be executed via npx, npm, or a local node_modules/.bin path.

    How it works
    - The shim uses environment overrides at runtime:
      - UVX_PATH: path to the uvx binary (default "uvx").
      - SVC_INFRA_REPO: overrides the repo embedded in the shim.
      - SVC_INFRA_REF: overrides the ref embedded in the shim.
      - UVX_REFRESH: if set (any value), adds --refresh to uvx args.
    - The repo argument is normalized to an HTTPS GitHub URL ending with .git. Supported forms: owner/repo, github:owner/repo, SSH (git@github.com:...), or HTTPS.

    Important parameters
    - tool_name: CLI name to publish (also the shim filename without extension).
    - module: Python module run with `python -m <module>`; must be a non-empty dotted path and must speak MCP over stdio when given --transport stdio.
    - repo/ref: Git source of the MCP server; ref defaults to "main".
    - package_json: location of package.json to update or create.
    - bin_dir: directory for the shim (default "mcp-shim/bin").
    - package_name: used only if creating a new package.json.
    - base_dir: optional root to resolve paths against; defaults to CWD.
    - force: overwrite an existing shim file.
    - dry_run: compute results and return emitted files without writing to disk.

    Returns
    - On success: { status: "ok", action: "created"|"updated"|"exists", tool_name, module, repo, ref, package_json, bin_path }.
    - On dry run: { status: "dry_run", files: { <package_json>: "...", <shim_path>: "..." }, ... }.
    - On error (e.g., read-only): { status: "error", error: <code>, message, suggestion? }.

    Goal
    - Make stdio-based MCP servers installable and runnable externally with a stable CLI (e.g., `npx <tool_name> ...`).
    """
    if not module or module.strip(". ") == "":
        return {"status": "error", "error": "invalid_module", "message": "module must be a non-empty dotted path"}

    repo_url = _normalize_repo(repo)

    # 3) build bin dir with preference for cleaned/inferred pkg root
    final_bin_dir = _coerce_bin_dir(bin_dir=bin_dir)

    return add_shim(
        tool_name=tool_name,
        module=module,
        repo=repo_url,
        ref=ref,
        package_json=Path(package_json),
        bin_dir=final_bin_dir,
        package_name=package_name,
        force=force,
        base_dir=Path(base_dir) if base_dir else None,
        dry_run=dry_run,
    )

def mcp_publish_remove(
        tool_name: str,
        package_json: str = "package.json",
        bin_dir: str = "mcp-shim/bin",
        delete_file: bool = False,
        base_dir: Optional[str] = None,
) -> dict:
    """
    Remove the CLI exposure for a stdio MCP server created by mcp_publish_add.

    What it does
    - Deletes the <tool_name> entry from package.json "bin" (if present).
    - Optionally deletes the generated shim file at <bin_dir>/<tool_name>.js when delete_file=True.
    - Paths are resolved relative to base_dir when provided; otherwise current working directory is used.

    Parameters
    - tool_name: CLI name whose mapping/shim should be removed.
    - package_json: path to package.json to update.
    - bin_dir: directory where the shim lives (default "mcp-shim/bin").
    - delete_file: also unlink the shim file.
    - base_dir: optional root to resolve paths against.

    Returns
    - { status: "ok", removed_from_package_json: bool, file_deleted: bool, shim_path, package_json }.
    - On error: { status: "error", error: "os_error", message }.

    Goal
    - Cleanly retract the external CLI for a stdio MCP server created by this tool.
    """
    final_bin_dir = _coerce_bin_dir(bin_dir=bin_dir)

    return remove_shim(
        tool_name=tool_name,
        package_json=Path(package_json),
        bin_dir=final_bin_dir,
        delete_file=delete_file,
        base_dir=Path(base_dir) if base_dir else None,
    )

def make_executable(targets: Union[str, Path, Iterable[Union[str, Path]]]) -> list[str]:
    """
    Ensure one or more files are marked executable (chmod +x equivalent).

    Args:
        targets: A path (string or Path) or iterable of paths.

    Returns:
        List of absolute paths that were updated (or confirmed executable).
    """
    if isinstance(targets, (str, Path)):
        targets = [targets]

    updated: list[str] = []
    for t in targets:
        p = Path(t).resolve()
        if not p.exists():
            raise FileNotFoundError(f"File does not exist: {p}")
        # Add executable bits (owner, group, others)
        mode = p.stat().st_mode
        p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        updated.append(str(p))
    return updated