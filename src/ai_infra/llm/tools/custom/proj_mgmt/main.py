from __future__ import annotations
import re, json, shutil, difflib
from pathlib import Path
from typing import Sequence, Literal

from ai_infra.llm.tools.custom.proj_mgmt.utils import (
    _REPO_ROOT, _confine, _tree, _walk, _read_small, _detect_tasks, _shim_cwd
)

import asyncio
import subprocess


async def project_scan(
        depth_tree: int = 5,
        include_capabilities: bool = True,
        include_git: bool = True,
        include_tasks: bool = True,
        include_env_keys: bool = True,
) -> str:
    """Async: Return concise JSON describing repo, tree, caps, git, tasks, env keys.
    Heavy IO is offloaded to a worker thread. Git calls are safe (no global chdir).
    """
    root = _REPO_ROOT

    def _run_scan() -> str:
        data: dict = {"repo_root": str(root)}
        data["tree"] = _tree(root, depth_tree)
        if include_capabilities:
            caps = []
            if (root / "pyproject.toml").exists(): caps.append("Python/Poetry")
            elif (root / "requirements.txt").exists(): caps.append("Python/pip")
            if (root / "package.json").exists(): caps.append("Node")
            if (root / "pom.xml").exists(): caps.append("Java/Maven")
            if any((root / f).exists() for f in ("build.gradle","build.gradle.kts")): caps.append("Java/Gradle")
            if any((root / f).exists() for f in ("Dockerfile","docker-compose.yml","compose.yml")): caps.append("Docker")
            for tool in ("poetry","npm","yarn","pnpm","mvn","gradle","docker","make","just","task","svc-infra"):
                if shutil.which(tool): caps.append(f"{tool} on PATH")
            data["capabilities"] = sorted(set(caps))
        if include_git:
            def _git(args: list[str]) -> str:
                try:
                    if not shutil.which("git"):
                        return ""
                    res = subprocess.run(["git", *args], cwd=str(root), text=True, capture_output=True)
                    return (res.stdout or "").strip()
                except Exception:
                    return ""
            branch = _git(["rev-parse","--abbrev-ref","HEAD"]) or ""
            upstream = _git(["rev-parse","--abbrev-ref","--symbolic-full-name","@{u}"]) or ""
            ahead_behind = ""
            if branch and upstream:
                ab = _git(["rev-list","--left-right","--count",f"{upstream}...HEAD"]) or ""
                if ab:
                    left,right = (ab.split() + ["0","0"])[:2]
                    ahead_behind = f"{right} ahead / {left} behind"
            recent = _git(["--no-pager","log","--oneline","-n","3"]) or ""
            remotes = _git(["remote","-v"]) or ""
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
                    k = line.split("=",1)[0].strip()
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
    """Async listing of files/dirs (repo-root sandboxed)."""
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
    """Async safe file read with size/preview guards."""
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
            lines = lines[: head_lines]
        if tail_lines is not None:
            lines = lines[-tail_lines:]
        body = "\n".join(lines)
        if truncated:
            body += "\n... [truncated]"
        return body

    return await asyncio.to_thread(_read)


WriteMode = Literal["write","append","replace","rename","delete","mkdir"]

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
    """Async multi-op writer, repo-sandboxed. All work happens in a thread."""
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
                new_text, n = re.subn(find, replace or "", text, count=0 if count is None else count, flags=re.MULTILINE)
            else:
                if count is None or count <= 0:
                    n = text.count(find)
                    new_text = text.replace(find, replace or "")
                else:
                    # limited literal replaces
                    parts = text.split(find)
                    new_text = (replace or "").join(parts[:count+1]) + find.join(parts[count+1:])
                    n = min(len(parts)-1, count)
            if new_text == text:
                return "[replace] no changes"
            if not overwrite and p.exists():
                # Make a simple .bak once per call to avoid accidental loss
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    bak.write_text(text, encoding="utf-8")
            p.write_text(new_text, encoding="utf-8")
            diff = "\n".join(difflib.unified_diff(text.splitlines(), new_text.splitlines(), lineterm=""))
            clipped = "\n".join(diff.splitlines()[:300])
            return f"[replace] {n} change(s)\n{clipped}"

        raise ValueError(f"Unknown mode: {mode}")

    return await asyncio.to_thread(_write)

__all__ = [
    "project_scan",
    "files_list",
    "file_read",
    "file_write",
]