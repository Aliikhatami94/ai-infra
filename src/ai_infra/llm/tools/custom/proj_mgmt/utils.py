from __future__ import annotations
import re, json, os
from pathlib import Path
from typing import Iterable, Sequence, Union

# ---------- Repo root & sandbox ----------

_ROOT_SIGNALS = (
    "pyproject.toml","package.json","pom.xml","build.gradle","build.gradle.kts",
    ".git","Makefile","Justfile","Taskfile.yml","Taskfile.yaml",
    "Dockerfile","docker-compose.yml","compose.yml",
)

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while True:
        if any((cur / s).exists() for s in _ROOT_SIGNALS):
            return cur
        if cur.parent == cur:
            return start.resolve()
        cur = cur.parent

_REPO_ROOT = Path(os.getenv("REPO_ROOT", os.getcwd())).resolve()

class ToolException(RuntimeError):
    pass

_CWD_PROC_PREFIXES = ("/proc/self/cwd",)  # extend if you need more shims later

def _normalize_user_path(p: Path) -> Path:
    s = str(p)
    for pref in _CWD_PROC_PREFIXES:
        if s == pref or s.startswith(pref + "/"):
            tail = s[len(pref):].lstrip("/")
            return (Path(os.getcwd()).resolve() / tail).resolve()
    return p

def _confine(path: Union[str, Path]) -> Path:
    """
    Map user-supplied path to a real path under _REPO_ROOT, handling proc/symlink cwd.
    Raises ToolException if it escapes the repo root.
    """
    root = _REPO_ROOT.resolve()
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
        raise ToolException(f"Path escapes repo root: {p}")

    return p

def _shim_cwd(path: str) -> str:
    if path.startswith("/proc/self/cwd"):
        return str(Path(os.getcwd()).resolve() / path.replace("/proc/self/cwd/", "", 1))
    return path

# ---------- Utils ----------

_IGNORED_DIRS = {
    ".git",".hg",".svn",".idea",".vscode","node_modules",".venv","venv",".tox",
    "dist","build","__pycache__",".pytest_cache",".mypy_cache",".next",".turbo",".cache",".gradle",
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

def _walk(
        root: Path, max_depth: int, exclude_globs: Sequence[str] | None
) -> Iterable[Path]:
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

def _tree(
        root: Path, max_depth: int, max_entries_per_dir: int = 80
) -> str:
    lines: list[str] = []
    def walk(d: Path, prefix: str, depth: int):
        try:
            entries = [p for p in sorted(d.iterdir(), key=lambda x:(not x.is_dir(), x.name.lower()))
                       if p.name not in _IGNORED_DIRS]
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
            scr = list(sorted((data.get("scripts") or {}).keys()))
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
            scr = list(sorted(((data.get("tool") or {}).get("poetry") or {}).get("scripts", {}).keys()))
            if scr:
                out["poetry"] = scr[:30]
        except Exception:
            pass
    return out