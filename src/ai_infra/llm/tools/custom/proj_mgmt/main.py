from __future__ import annotations

import asyncio
import difflib
import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from ai_infra.llm.tools.custom.proj_mgmt.utils import (
    _REPO_ROOT,
    _confine,
    _detect_tasks,
    _read_small,
    _shim_cwd,
    _tree,
    _walk,
)


async def project_scan(
    depth_tree: int = 5,
    include_capabilities: bool = True,
    include_git: bool = True,
    include_tasks: bool = True,
    include_env_keys: bool = True,
) -> str:
    """Scan the project repository and return comprehensive metadata.

    Returns JSON describing the repository structure, capabilities, git status,
    available tasks, and environment variable keys. Use this to understand the
    project before making changes.

    Args:
        depth_tree: Maximum depth for directory tree (default 5). Higher values
            show more nested files but take longer.
        include_capabilities: Include detected project capabilities like
            "Python/Poetry", "Node", "Docker" (default True).
        include_git: Include git metadata like branch, upstream, recent commits
            (default True).
        include_tasks: Include detected tasks from Makefile, package.json scripts,
            etc. (default True).
        include_env_keys: Include environment variable names from .env file,
            without values for security (default True).

    Returns:
        JSON string with project metadata. Always returns a string, never errors.
    """
    root = _REPO_ROOT

    def _run_scan() -> str:
        data: dict = {"repo_root": str(root)}
        data["tree"] = _tree(root, depth_tree)
        if include_capabilities:
            caps = []
            if (root / "pyproject.toml").exists():
                caps.append("Python/Poetry")
            elif (root / "requirements.txt").exists():
                caps.append("Python/pip")
            if (root / "package.json").exists():
                caps.append("Node")
            if (root / "pom.xml").exists():
                caps.append("Java/Maven")
            if any((root / f).exists() for f in ("build.gradle", "build.gradle.kts")):
                caps.append("Java/Gradle")
            if any(
                (root / f).exists() for f in ("Dockerfile", "docker-compose.yml", "compose.yml")
            ):
                caps.append("Docker")
            for tool in (
                "poetry",
                "npm",
                "yarn",
                "pnpm",
                "mvn",
                "gradle",
                "docker",
                "make",
                "just",
                "task",
                "svc-infra",
            ):
                if shutil.which(tool):
                    caps.append(f"{tool} on PATH")
            data["capabilities"] = sorted(set(caps))
        if include_git:

            def _git(args: list[str]) -> str:
                try:
                    if not shutil.which("git"):
                        return ""
                    res = subprocess.run(
                        ["git", *args], cwd=str(root), text=True, capture_output=True
                    )
                    return (res.stdout or "").strip()
                except Exception:
                    return ""

            branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or ""
            upstream = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]) or ""
            ahead_behind = ""
            if branch and upstream:
                ab = _git(["rev-list", "--left-right", "--count", f"{upstream}...HEAD"]) or ""
                if ab:
                    left, right = (ab.split() + ["0", "0"])[:2]
                    ahead_behind = f"{right} ahead / {left} behind"
            recent = _git(["--no-pager", "log", "--oneline", "-n", "3"]) or ""
            remotes = _git(["remote", "-v"]) or ""
            data["git"] = {
                "branch": branch,
                "upstream": upstream,
                "ahead_behind": ahead_behind,
                "recent_commits": recent,
                "remotes": remotes,
            }
        if include_tasks:
            data["tasks"] = _detect_tasks(root)
        if include_env_keys:
            keys = []
            f = root / ".env"
            if f.exists():
                for line in f.read_text(errors="ignore").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k = line.split("=", 1)[0].strip()
                    if re.fullmatch(r"[A-Z0-9_]+", k):
                        keys.append(k)
            data["env_keys"] = keys[:50]
        return json.dumps(data, ensure_ascii=False)

    return await asyncio.to_thread(_run_scan)


async def files_list(
    root_or_dir: str = ".",
    glob: str | None = None,
    exclude: Sequence[str] | None = None,
    max_depth: int = 4,
    as_tree: bool = False,
    limit: int = 1000,
) -> str:
    """List files and directories in the project workspace.

    All paths are sandboxed to the workspace root. Use this to explore the file
    structure before reading or writing files.

    Args:
        root_or_dir: Starting directory, relative to workspace root (default ".").
            Examples: ".", "src", "src/components".
        glob: Optional glob pattern to filter files. Examples: "*.py", "**/*.ts",
            "src/**/*.json". If None, lists all files up to max_depth.
        exclude: Patterns to exclude. Examples: ["__pycache__", "node_modules", "*.pyc"].
        max_depth: Maximum directory depth to traverse (default 4).
        as_tree: If True, return indented tree format. If False, return flat list
            with one path per line (default False).
        limit: Maximum number of files to return (default 1000). Prevents
            overwhelming output in large projects.

    Returns:
        Newline-separated list of paths, or indented tree if as_tree=True.
        Paths are relative to workspace root.
    """
    base = _confine(_shim_cwd(root_or_dir))

    def _list() -> str:
        if as_tree:
            return _tree(base, max_depth=max_depth)
        # flat list (glob or walk)
        paths: list[Path] = []
        if glob:
            for p in base.glob(glob):
                try:
                    p.relative_to(_REPO_ROOT)  # enforce sandbox
                    if exclude and any(p.match(g) or p.name == g for g in (exclude or [])):
                        continue
                    paths.append(p)
                    if len(paths) >= limit:
                        break
                except Exception:
                    continue
        else:
            for p in _walk(base, max_depth=max_depth, exclude_globs=exclude or []):
                paths.append(p)
                if len(paths) >= limit:
                    break
        return "\n".join(str(p.relative_to(_REPO_ROOT)) for p in paths)

    return await asyncio.to_thread(_list)


async def file_read(
    path: str,
    *,
    max_bytes: int = 200_000,
    head_lines: int | None = None,
    tail_lines: int | None = None,
    binary_hex: bool = True,
) -> str:
    """Read a file from the project workspace.

    All paths are sandboxed to the workspace root. Large files are automatically
    truncated with a note. Binary files can be previewed as hex.

    Args:
        path: File path relative to workspace root. Examples: "README.md",
            "src/main.py", "config/settings.json".
        max_bytes: Maximum bytes to read (default 200,000). Files larger than
            this are truncated with a "[truncated]" note.
        head_lines: If set, only return the first N lines of the file.
        tail_lines: If set, only return the last N lines of the file.
        binary_hex: If True and file is binary, return hex preview (default True).
            If False, return "[binary preview omitted]" message.

    Returns:
        File contents as string, possibly with truncation note.
        For binary files: hex preview or omitted message.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    p = _confine(_shim_cwd(path))

    def _read() -> str:
        if not p.exists():
            raise FileNotFoundError(str(p))
        content, truncated = _read_small(p, max_bytes=max_bytes)
        if isinstance(content, bytes):
            if not binary_hex:
                return f"[binary {p.name} size<= {max_bytes} bytes preview omitted]"
            h = content[:2048].hex()
            note = " (truncated)" if truncated else ""
            return f"[binary hex{note}] {h}"
        # text
        lines = content.splitlines()
        if head_lines is not None:
            lines = lines[:head_lines]
        if tail_lines is not None:
            lines = lines[-tail_lines:]
        body = "\n".join(lines)
        if truncated:
            body += "\n... [truncated]"
        return body

    return await asyncio.to_thread(_read)


WriteMode = Literal["write", "append", "replace", "rename", "delete", "mkdir"]


async def file_write(
    mode: WriteMode,
    *,
    path: str,
    content: str | None = None,
    create_dirs: bool = True,
    overwrite: bool = False,
    find: str | None = None,
    replace: str | None = None,
    regex: bool = False,
    count: int | None = None,
    new_path: str | None = None,
    make_parents: bool | None = None,  # alias for create_dirs
) -> str:
    """Write, modify, or delete a file in the project workspace.

    All paths are sandboxed to the workspace root. Supports multiple operations:
    write, append, replace, rename, delete, and mkdir.

    Args:
        mode: Operation to perform:
            - "write": Create or overwrite a file with content
            - "append": Add content to end of file (creates if missing)
            - "replace": Find and replace text within existing file
            - "rename": Move/rename a file (requires new_path)
            - "delete": Remove a file (empty dirs only)
            - "mkdir": Create a directory
        path: File/directory path relative to workspace root.
        content: Text content for write/append modes.
        create_dirs: Create parent directories if missing (default True).
        overwrite: Allow overwriting existing files (default False).
            Required for write mode on existing files.
        find: String or regex to find in replace mode (required for replace).
        replace: Replacement text in replace mode (empty string if None).
        regex: Treat find as regex pattern instead of literal (default False).
        count: Max replacements in replace mode (None = replace all).
        new_path: Destination path for rename mode (required for rename).
        make_parents: Alias for create_dirs (for compatibility).

    Returns:
        Success message describing the operation, e.g. "[write] path (123 bytes)".
        For replace mode, includes a diff of changes.

    Raises:
        FileExistsError: If overwriting without overwrite=True.
        FileNotFoundError: If file missing for replace mode.
        ValueError: If required args missing for mode.
        IsADirectoryError: If deleting non-empty directory.
    """
    if make_parents is not None:
        create_dirs = make_parents

    p = _confine(_shim_cwd(path))

    def _write() -> str:
        if mode == "mkdir":
            if p.exists():
                return f"[mkdir] exists: {p}"
            p.mkdir(parents=create_dirs, exist_ok=False)
            return f"[mkdir] created: {p}"

        if mode == "delete":
            if not p.exists():
                return f"[delete] not found: {p}"
            if p.is_dir():
                # safety: only remove empty dirs
                if any(p.iterdir()):
                    raise IsADirectoryError(f"Directory not empty: {p}")
                p.rmdir()
            else:
                p.unlink()
            return f"[delete] removed: {p}"

        if mode == "rename":
            if not new_path:
                raise ValueError("new_path is required for rename")
            dst = _confine(new_path)
            if not overwrite and dst.exists():
                raise FileExistsError(f"Target exists: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            p.replace(dst)
            return f"[rename] {p} -> {dst}"

        if mode == "write":
            if p.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite without overwrite=True: {p}")
            if create_dirs:
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content or "", encoding="utf-8")
            return f"[write] {p} ({len(content or '')} bytes)"

        if mode == "append":
            if create_dirs:
                p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(content or "")
            return f"[append] {p} (+{len(content or '')} bytes)"

        if mode == "replace":
            if not p.exists():
                raise FileNotFoundError(str(p))
            text = p.read_text(encoding="utf-8", errors="replace")
            if find is None:
                raise ValueError("find is required for replace")
            if regex:
                new_text, n = re.subn(
                    find,
                    replace or "",
                    text,
                    count=0 if count is None else count,
                    flags=re.MULTILINE,
                )
            else:
                if count is None or count <= 0:
                    n = text.count(find)
                    new_text = text.replace(find, replace or "")
                else:
                    # limited literal replaces
                    parts = text.split(find)
                    new_text = (replace or "").join(parts[: count + 1]) + find.join(
                        parts[count + 1 :]
                    )
                    n = min(len(parts) - 1, count)
            if new_text == text:
                return "[replace] no changes"
            if not overwrite and p.exists():
                # Make a simple .bak once per call to avoid accidental loss
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    bak.write_text(text, encoding="utf-8")
            p.write_text(new_text, encoding="utf-8")
            diff = "\n".join(
                difflib.unified_diff(text.splitlines(), new_text.splitlines(), lineterm="")
            )
            clipped = "\n".join(diff.splitlines()[:300])
            return f"[replace] {n} change(s)\n{clipped}"

        raise ValueError(f"Unknown mode: {mode}")

    return await asyncio.to_thread(_write)


__all__ = [
    "file_read",
    "file_write",
    "files_list",
    "project_scan",
]
