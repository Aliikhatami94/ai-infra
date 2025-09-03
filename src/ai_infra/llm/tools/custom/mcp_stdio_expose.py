# ai_infra/llm/tools/custom/mcp_stdio_expose.py
from __future__ import annotations
import re
from pathlib import Path, PurePosixPath
from typing import Optional
from ai_infra.mcp.expose.core import add_shim, remove_shim

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

def _coerce_bin_dir(bin_dir: Optional[str], python_package_root: Optional[str]) -> Path:
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

    Robust params:
    - `repo` accepts owner/repo, github:owner/repo, SSH, or HTTPS and is normalized to HTTPS .git
    - `python_package_root` may be a path (e.g. 'src/pkg'); we clean it to just 'pkg'
    - If pkg root omitted, we infer from `module` and use it if 'src/<pkg>' exists
    - If both pkg root and bin_dir are provided, pkg root wins

    Read-only:
    - Pass `base_dir` (writable root) or `dry_run=True` (emit files; no writes)
    """
    if not module or module.strip(". ") == "":
        return {"status": "error", "error": "invalid_module", "message": "module must be a non-empty dotted path"}

    repo_url = _normalize_repo(repo)

    # 1) clean explicit python_package_root (path → package name)
    cleaned_pkg_root = _clean_pkg_root(python_package_root)

    # 2) if no explicit root, infer from module and prefer when src/<pkg> exists
    if not cleaned_pkg_root:
        inferred = _infer_pkg_root(module)
        if inferred and (Path("src") / inferred).exists():
            cleaned_pkg_root = inferred

    # 3) build bin dir with preference for cleaned/inferred pkg root
    final_bin_dir = _coerce_bin_dir(bin_dir=bin_dir, python_package_root=cleaned_pkg_root)

    return add_shim(
        tool_name=tool_name,
        module=module,
        repo=repo_url,
        ref=ref,
        package_json=Path(package_json),
        bin_dir=final_bin_dir,
        python_package_root=cleaned_pkg_root,
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
    If pkg root was used when adding, pass the same `python_package_root` (path accepted; will be cleaned).
    """
    cleaned_pkg_root = _clean_pkg_root(python_package_root)
    final_bin_dir = _coerce_bin_dir(bin_dir=bin_dir, python_package_root=cleaned_pkg_root)

    return remove_shim(
        tool_name=tool_name,
        package_json=Path(package_json),
        bin_dir=final_bin_dir,
        python_package_root=cleaned_pkg_root,
        delete_file=delete_file,
        base_dir=Path(base_dir) if base_dir else None,
    )