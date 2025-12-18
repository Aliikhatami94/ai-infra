from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from pathlib import Path

# ---------- Repo root & sandbox ----------

_ROOT_SIGNALS = (
    "pyproject.toml",
    "package.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    ".git",
    "Makefile",
    "Justfile",
    "Taskfile.yml",
    "Taskfile.yaml",
    "Dockerfile",
    "docker-compose.yml",
    "compose.yml",
)


def _find_repo_root(start: Path) -> Path:
    """Walk up from start until we find a project marker file."""
    cur = start.resolve()
    while True:
        if any((cur / s).exists() for s in _ROOT_SIGNALS):
            return cur
        if cur.parent == cur:
            return start.resolve()
        cur = cur.parent


# Default: auto-detect from env var or cwd
_DEFAULT_ROOT = Path(os.getenv("REPO_ROOT", os.getcwd())).resolve()

# Context variable for per-request/per-agent workspace override
_workspace_root: ContextVar[Path | None] = ContextVar("workspace_root", default=None)


def _set_workspace_root(path: str | Path | None) -> None:
    """Internal: Set workspace root without deprecation warning.

    Used by Workspace.configure_proj_mgmt() to set the root.
    """
    if path is None:
        _workspace_root.set(None)
    else:
        _workspace_root.set(Path(path).resolve())


def set_workspace_root(path: str | Path | None) -> None:
    """Set an explicit workspace root for the current context.

    .. deprecated:: 1.0
        Use Agent(workspace=...) instead. Direct workspace configuration
        will be removed in a future version.

    Use this to sandbox tools to a specific directory instead of
    auto-detecting the repo root.

    Args:
        path: The workspace directory to sandbox to.
              Pass None to reset to auto-detection mode.

    Example:
        # Repo-sandboxed (default): auto-detects project root
        await file_read("src/main.py")

        # Workspace-sandboxed: explicit directory
        set_workspace_root("/tmp/agent-workspace")
        await file_read("input.txt")  # reads /tmp/agent-workspace/input.txt

        # Reset to auto-detection
        set_workspace_root(None)
    """
    import warnings

    warnings.warn(
        "set_workspace_root() is deprecated. Use Agent(workspace=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _set_workspace_root(path)


def get_workspace_root() -> Path:
    """Get the current workspace root (explicit or auto-detected)."""
    override = _workspace_root.get()
    if override is not None:
        return override
    return _DEFAULT_ROOT


# Backward compatibility alias
_REPO_ROOT = _DEFAULT_ROOT


class ToolException(RuntimeError):
    pass


_CWD_PROC_PREFIXES = ("/proc/self/cwd",)  # extend if you need more shims later


def _normalize_user_path(p: Path) -> Path:
    s = str(p)
    for pref in _CWD_PROC_PREFIXES:
        if s == pref or s.startswith(pref + "/"):
            tail = s[len(pref) :].lstrip("/")
            return (Path(os.getcwd()).resolve() / tail).resolve()
    return p


def _confine(path: str | Path, *, workspace: Path | None = None) -> Path:
    """
    Map user-supplied path to a real path under the workspace root.

    ⚠️ SECURITY: This function is the primary defense against path traversal attacks.
    ALL user-supplied paths MUST be processed through this function before any
    filesystem operation. This prevents attacks like "../../etc/passwd".

    The function:
    1. Resolves the path to an absolute path (following symlinks)
    2. Verifies the resolved path is under the workspace root
    3. Raises ToolException if the path escapes the sandbox

    Args:
        path: User-supplied path (relative or absolute)
        workspace: Optional explicit workspace root. If None, uses
                   get_workspace_root() (context var or default).

    Returns:
        Resolved absolute path guaranteed to be under the workspace root.

    Raises:
        ToolException: If path escapes the workspace root (e.g., "../../etc/passwd").

    Examples:
        >>> _confine("src/main.py")  # OK: relative path
        PosixPath('/workspace/src/main.py')

        >>> _confine("../../../etc/passwd")  # BLOCKED
        ToolException: Path escapes workspace root: /etc/passwd
    """
    root = (workspace or get_workspace_root()).resolve()
    p = _normalize_user_path(Path(path))

    # Make path absolute under root when relative; always resolve to realpath
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()

    try:
        # Will raise ValueError if p is not under root
        p.relative_to(root)
    except Exception:
        raise ToolException(f"Path escapes workspace root: {p}")

    return p


def _shim_cwd(path: str) -> str:
    if path.startswith("/proc/self/cwd"):
        return str(Path(os.getcwd()).resolve() / path.replace("/proc/self/cwd/", "", 1))
    return path


# ---------- Utils ----------

_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".next",
    ".turbo",
    ".cache",
    ".gradle",
}


def _is_text_bytes(b: bytes) -> bool:
    if not b:
        return True
    # Heuristic: reject NULs and many non-printables
    if b"\x00" in b:
        return False
    # Allow common UTF BOMs
    sample = b[:1024]
    try:
        sample.decode("utf-8")
        return True
    except Exception:
        return False


def _read_small(path: Path, max_bytes: int) -> tuple[str | bytes, bool]:
    data = path.read_bytes()
    truncated = len(data) > max_bytes > 0
    if truncated:
        data = data[:max_bytes]
    is_text = _is_text_bytes(data)
    return (data.decode("utf-8", errors="replace") if is_text else data, truncated)


def _walk(root: Path, max_depth: int, exclude_globs: Sequence[str] | None) -> Iterable[Path]:
    root = root.resolve()
    if max_depth < 0:
        return
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            for p in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                name = p.name
                if name in _IGNORED_DIRS:
                    continue
                if exclude_globs and any(p.match(g) or name == g for g in exclude_globs):
                    continue
                yield p
                if p.is_dir() and depth < max_depth:
                    stack.append((p, depth + 1))
        except Exception:
            continue


def _tree(root: Path, max_depth: int, max_entries_per_dir: int = 80) -> str:
    lines: list[str] = []

    def walk(d: Path, prefix: str, depth: int):
        try:
            entries = [
                p
                for p in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                if p.name not in _IGNORED_DIRS
            ]
        except Exception:
            return
        shown = 0
        n = len(entries)
        for i, p in enumerate(entries, 1):
            if shown >= max_entries_per_dir:
                lines.append(f"{prefix}└── … ({n - i + 1} more)")
                break
            connector = "└──" if i == n else "├──"
            label = p.name + ("/" if p.is_dir() else "")
            lines.append(f"{prefix}{connector} {label}")
            shown += 1
            if p.is_dir() and depth > 0:
                child_prefix = f"{prefix}{'    ' if i == n else '│   '}"
                walk(p, child_prefix, depth - 1)

    lines.append(root.name + "/")
    walk(root, "", max_depth)
    return "\n".join(lines)


def _detect_tasks(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    # Makefile
    mf = root / "Makefile"
    if mf.exists():
        try:
            names = []
            for line in mf.read_text(errors="ignore").splitlines():
                m = re.match(r"^([A-Za-z0-9._-]+)\s*:", line)
                if m and not m.group(1).startswith("."):
                    if m.group(1) not in names:
                        names.append(m.group(1))
            if names:
                out["make"] = names[:30]
        except Exception:
            pass
    # package.json scripts
    pj = root / "package.json"
    if pj.exists():
        try:
            data = json.loads(pj.read_text(errors="ignore"))
            scr = sorted((data.get("scripts") or {}).keys())
            if scr:
                out["npm"] = scr[:30]
        except Exception:
            pass
    # poetry scripts
    pp = root / "pyproject.toml"
    if pp.exists():
        try:
            import tomllib

            data = tomllib.loads(pp.read_text(errors="ignore"))
            scr = sorted(((data.get("tool") or {}).get("poetry") or {}).get("scripts", {}).keys())
            if scr:
                out["poetry"] = scr[:30]
        except Exception:
            pass
    return out
