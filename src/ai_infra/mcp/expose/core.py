from __future__ import annotations
import errno, json, os, stat
from pathlib import Path
from typing import Dict, Optional, Tuple

JS_TEMPLATE_UVX_MODULE = """#!/usr/bin/env node
const {{ spawn }} = require("child_process");

// Config from env with sane defaults
const UVX  = process.env.UVX_PATH || "uvx";
const REPO = process.env.SVC_INFRA_REPO || "{repo}";
const REF  = process.env.SVC_INFRA_REF  || "{ref}";
const SPEC = `git+${{REPO}}@${{REF}}`;

// Run: uvx --from SPEC python -m <module> --transport stdio <passthrough-args>
const args = [
  "--quiet",
  ...(process.env.UVX_REFRESH ? ["--refresh"] : []),
  "--from", SPEC,
  "python", "-m", "{py_module}",
  "--transport", "stdio",
  ...process.argv.slice(2)
];

const child = spawn(UVX, args, {{ stdio: "inherit", shell: process.platform === "win32" }});
child.on("exit", code => process.exit(code));
"""

def _load_json(p: Path) -> Dict:
    return json.loads(p.read_text()) if p.exists() else {}

def _dump_json(p: Path, data: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")

def _ensure_executable(p: Path) -> None:
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def _resolve_paths(
        *,
        base_dir: Optional[Path],
        package_json: Path,
        bin_dir: Path,
        tool_name: str,
) -> Tuple[Path, Path]:
    root = base_dir.resolve() if base_dir else Path.cwd()
    package_json = (root / package_json).resolve()
    bin_dir = (root / bin_dir).resolve()
    shim_path = bin_dir / f"{tool_name}.js"
    return package_json, shim_path

def add_shim(
        *,
        tool_name: str,
        module: str,
        repo: str,
        ref: str = "main",
        package_json: Path = Path("package.json"),
        bin_dir: Path = Path("mcp-shim") / "bin",
        package_name: str = "mcp-shims",
        force: bool = False,
        base_dir: Optional[Path] = None,
        dry_run: bool = False,
) -> Dict:
    try:
        package_json, shim_path = _resolve_paths(
            base_dir=base_dir,
            package_json=package_json,
            bin_dir=bin_dir,
            tool_name=tool_name,
        )
        js = JS_TEMPLATE_UVX_MODULE.format(repo=repo, ref=ref, py_module=module)

        # compute relative path for package.json -> "bin" map
        root_dir = package_json.parent
        try:
            rel_for_bin = shim_path.relative_to(root_dir).as_posix()
        except ValueError:
            # fallback (different drives, etc.)
            rel_for_bin = os.path.relpath(shim_path.as_posix(), root_dir.as_posix())

        if dry_run:
            pkg = _load_json(package_json)
            if not pkg:
                pkg = {"name": package_name, "version": "0.0.0", "private": True, "bin": {}}
            if "bin" not in pkg or not isinstance(pkg["bin"], dict):
                pkg["bin"] = {}
            pkg["bin"][tool_name] = rel_for_bin
            return {
                "status": "dry_run",
                "action": "emit",
                "tool_name": tool_name,
                "module": module,
                "repo": repo,
                "ref": ref,
                "package_json": str(package_json),
                "bin_path": rel_for_bin,
                "files": {
                    str(package_json): json.dumps(pkg, indent=2) + "\n",
                    str(shim_path): js,
                },
                "hint": "Filesystem was not modified. Apply these changes in a writable workspace.",
            }

        # Real writes
        pkg = _load_json(package_json)
        if not pkg:
            pkg = {"name": package_name, "version": "0.0.0", "private": True, "bin": {}}
        if "bin" not in pkg or not isinstance(pkg["bin"], dict):
            pkg["bin"] = {}

        action = "exists"
        if not shim_path.exists() or force:
            shim_path.parent.mkdir(parents=True, exist_ok=True)
            shim_path.write_text(js)
            _ensure_executable(shim_path)
            action = "created" if not force else "updated"

        pkg["bin"][tool_name] = rel_for_bin
        _dump_json(package_json, pkg)

        return {
            "status": "ok",
            "action": action,
            "tool_name": tool_name,
            "module": module,
            "repo": repo,
            "ref": ref,
            "package_json": str(package_json),
            "bin_path": rel_for_bin,
        }

    except OSError as e:
        readonly = e.errno in (errno.EROFS, errno.EACCES)
        return {
            "status": "error",
            "error": "read_only_filesystem" if readonly else "os_error",
            "message": str(e),
            "tool_name": tool_name,
            "package_json": str(package_json),
            "bin_dir": str(bin_dir),
            "base_dir": str(base_dir) if base_dir else None,
            "suggestion": (
                "Pass a writable base_dir or set dry_run=True and apply emitted files in a writable repo."
                if readonly else "Check paths/permissions."
            ),
        }

def remove_shim(
        *,
        tool_name: str,
        package_json: Path = Path("package.json"),
        bin_dir: Path = Path("mcp-shim") / "bin",
        delete_file: bool = False,
        base_dir: Optional[Path] = None,
) -> Dict:
    try:
        package_json, shim_path = _resolve_paths(
            base_dir=base_dir,
            package_json=package_json,
            bin_dir=bin_dir,
            tool_name=tool_name,
        )

        pkg = _load_json(package_json)
        removed = False
        if "bin" in pkg and tool_name in pkg["bin"]:
            del pkg["bin"][tool_name]
            _dump_json(package_json, pkg)
            removed = True

        file_deleted = False
        if delete_file and shim_path.exists():
            shim_path.unlink()
            file_deleted = True

        return {
            "status": "ok",
            "removed_from_package_json": removed,
            "file_deleted": file_deleted,
            "shim_path": str(shim_path),
            "package_json": str(package_json),
        }

    except OSError as e:
        return {
            "status": "error",
            "error": "os_error",
            "message": str(e),
            "tool_name": tool_name,
            "package_json": str(package_json),
        }