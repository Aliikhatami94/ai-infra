from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from ai_infra.mcp.expose.core import add_shim, remove_shim

_GH_SSH = re.compile(r"^git@github\.com:(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+)(?:\.git)?$")
_GH_HTTPS = re.compile(r"^https?://github\.com/(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+)(?:\.git)?$")
_GH_SHORT = re.compile(r"^(?:github:)?(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+)$")

def _ensure_git_suffix(url: str) -> str:
    return url if url.endswith(".git") else f"{url}.git"

def _normalize_repo(repo: str) -> str:
    """Accepts owner/repo, github:owner/repo, SSH, or HTTPS and returns full HTTPS .git URL."""
    repo = repo.strip()

    # HTTPS github.com
    m = _GH_HTTPS.match(repo)
    if m:
        return _ensure_git_suffix(f"https://github.com/{m.group('owner')}/{m.group('repo')}")

    # SSH github.com
    m = _GH_SSH.match(repo)
    if m:
        return _ensure_git_suffix(f"https://github.com/{m.group('owner')}/{m.group('repo')}")

    # Short github form
    m = _GH_SHORT.match(repo)
    if m:
        return _ensure_git_suffix(f"https://github.com/{m.group('owner')}/{m.group('repo')}")

    # Other schemes/hosts: leave as-is
    return repo

def _infer_pkg_root(module: str) -> Optional[str]:
    """Best-effort infer package root from module path (e.g., 'svc_infra.auth.mcp' -> 'svc_infra')."""
    head = module.split(".", 1)[0].strip()
    return head or None

def _coerce_bin_dir(
        bin_dir: Optional[str],
        python_package_root: Optional[str],
) -> Path:
    """Prefer python_package_root; fall back to provided bin_dir or default."""
    if python_package_root:
        return Path("src") / python_package_root / "mcp-shim" / "bin"
    if bin_dir:
        return Path(bin_dir)
    return Path("src") / "mcp-shim" / "bin"

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

    Robust param handling:
    - `repo` may be 'owner/repo', 'github:owner/repo', SSH, or HTTPS; it is normalized.
    - If `python_package_root` is omitted, we infer it from `module` and prefer
      `src/<pkg>/mcp-shim/bin` when `src/<pkg>` exists; otherwise we fall back to `src/mcp-shim/bin`.
    - If both `python_package_root` and `bin_dir` are provided but conflict, we prefer
      `python_package_root`.

    Read-only environments:
    - Pass `base_dir` to write under a known writable root, or
    - Set `dry_run=True` to return file contents without touching disk (the core also
      auto-falls back on EROFS/EACCES if you kept that behavior).
    """
    repo_url = _normalize_repo(repo)

    # Prefer explicit python_package_root; else infer from module if src/<pkg> exists.
    pkg_root = python_package_root
    if not pkg_root:
        inferred = _infer_pkg_root(module)
        if inferred and (Path("src") / inferred).exists():
            pkg_root = inferred

    # Build bin_dir with preference for pkg_root.
    final_bin_dir = _coerce_bin_dir(bin_dir=bin_dir, python_package_root=pkg_root)

    return add_shim(
        tool_name=tool_name,
        module=module,
        repo=repo_url,
        ref=ref,
        package_json=Path(package_json),
        bin_dir=final_bin_dir,
        python_package_root=pkg_root,
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
    If a package root was used when adding, pass the same `python_package_root`.
    """
    # Keep same bin_dir logic as add() to target the right path by default
    final_bin_dir = _coerce_bin_dir(bin_dir=bin_dir, python_package_root=python_package_root)

    return remove_shim(
        tool_name=tool_name,
        package_json=Path(package_json),
        bin_dir=final_bin_dir,
        python_package_root=python_package_root,
        delete_file=delete_file,
        base_dir=Path(base_dir) if base_dir else None,
    )