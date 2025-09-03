from __future__ import annotations
import json
from pathlib import Path
import stat
from typing import Dict, Optional

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

def add_shim(
        *,
        tool_name: str,
        module: str,
        repo: str,
        ref: str = "main",
        package_json: Path = Path("package.json"),
        bin_dir: Path = Path("src") / "mcp-shim" / "bin",
        python_package_root: Optional[str] = None,
        package_name: str = "mcp-shims",
        force: bool = False,
) -> Dict:
    # resolve bin dir
    if python_package_root:
        bin_dir = Path("src") / python_package_root / "mcp-shim" / "bin"

    # package.json (create or update)
    pkg = _load_json(package_json)
    if not pkg:
        pkg = {"name": package_name, "version": "0.0.0", "private": True, "bin": {}}
    if "bin" not in pkg or not isinstance(pkg["bin"], dict):
        pkg["bin"] = {}

    # shim file
    shim_path = bin_dir / f"{tool_name}.js"
    if not shim_path.exists() or force:
        js = JS_TEMPLATE_UVX_MODULE.format(repo=repo, ref=ref, py_module=module)
        shim_path.parent.mkdir(parents=True, exist_ok=True)
        shim_path.write_text(js)
        _ensure_executable(shim_path)
        action = "created" if not force else "updated"
    else:
        action = "exists"

    # bin map
    rel = shim_path.as_posix()
    pkg["bin"][tool_name] = rel
    _dump_json(package_json, pkg)

    return {
        "status": "ok",
        "action": action,
        "tool_name": tool_name,
        "module": module,
        "repo": repo,
        "ref": ref,
        "package_json": str(package_json),
        "bin_path": rel,
    }

def remove_shim(
        *,
        tool_name: str,
        package_json: Path = Path("package.json"),
        bin_dir: Path = Path("src") / "mcp-shim" / "bin",
        python_package_root: Optional[str] = None,
        delete_file: bool = False,
) -> Dict:
    if python_package_root:
        bin_dir = Path("src") / python_package_root / "mcp-shim" / "bin"
    shim_path = bin_dir / f"{tool_name}.js"

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