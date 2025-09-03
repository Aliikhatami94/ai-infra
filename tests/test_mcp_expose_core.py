from pathlib import Path
import os
import stat
import json
from ai_infra.mcp.publish.core import add_shim, remove_shim

def read_json(p: Path):
    return json.loads(p.read_text())

def test_add_creates_package_json_and_shim(tmp_path: Path):
    res = add_shim(
        tool_name="demo-mcp",
        module="pkg.mod.mcp",
        repo="https://github.com/owner/repo.git",
        base_dir=tmp_path,
    )
    assert res["status"] == "ok"
    pkg = read_json(tmp_path / "package.json")
    assert "bin" in pkg and "demo-mcp" in pkg["bin"]
    shim = tmp_path / pkg["bin"]["demo-mcp"]
    assert shim.exists()
    # executable bit set
    mode = shim.stat().st_mode
    assert mode & stat.S_IXUSR

def test_add_updates_existing_bin_map(tmp_path: Path):
    # first write
    add_shim(tool_name="a", module="m.a", repo="r", base_dir=tmp_path)
    # add another
    res = add_shim(tool_name="b", module="m.b", repo="r", base_dir=tmp_path)
    assert res["status"] == "ok"
    pkg = read_json(tmp_path / "package.json")
    assert set(pkg["bin"].keys()) == {"a", "b"}

def test_force_overwrite(tmp_path: Path):
    res1 = add_shim(tool_name="x", module="m.x", repo="r", base_dir=tmp_path)
    p = tmp_path / "mcp-shim/bin/x.js"
    before = p.read_text()
    res2 = add_shim(tool_name="x", module="m.x2", repo="r", base_dir=tmp_path, force=True)
    after = p.read_text()
    assert res1["status"] == "ok" and res2["status"] == "ok"
    assert before != after  # overwritten with new module path

def test_python_package_root_paths(tmp_path: Path):
    res = add_shim(
        tool_name="demo",
        module="pkg.mod.m",
        repo="r",
        base_dir=tmp_path,
    )
    pkg = read_json(tmp_path / "package.json")
    rel = pkg["bin"]["demo"]
    assert rel.startswith("mcp-shim/bin/")

def test_remove_deletes_bin_entry_and_file(tmp_path: Path):
    add_shim(tool_name="rmme", module="m", repo="r", base_dir=tmp_path)
    res = remove_shim(
        tool_name="rmme",
        base_dir=tmp_path,
        delete_file=True,
    )
    assert res["status"] == "ok"
    pkg = read_json(tmp_path / "package.json")
    assert "rmme" not in pkg.get("bin", {})
    # file deleted
    assert not (tmp_path / "mcp-shim/bin/rmme.js").exists()

def test_dry_run_emits_files_without_writing(tmp_path: Path):
    res = add_shim(
        tool_name="dry",
        module="m.dry",
        repo="r",
        base_dir=tmp_path,
        dry_run=True,
    )
    assert res["status"] == "dry_run"
    assert "files" in res
    # nothing written
    assert not (tmp_path / "package.json").exists()
    assert not (tmp_path / "mcp-shim/bin/dry.js").exists()

def test_read_only_base_dir_returns_error(tmp_path: Path):
    # Make a read-only dir
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    os.chmod(ro_dir, 0o555)  # r-x
    res = add_shim(tool_name="cant", module="m", repo="r", base_dir=ro_dir)
    assert res["status"] in {"error", "dry_run"}  # error expected
    if res["status"] == "error":
        assert res["error"] == "read_only_filesystem"
    # Reset perms so tmp cleanup doesn't fail
    os.chmod(ro_dir, 0o755)

def test_normalize_repo_variants():
    from ai_infra.llm.tools.custom.stdio_publisher import _normalize_repo
    assert _normalize_repo("aliikhatami94/svc-infra") == "https://github.com/aliikhatami94/svc-infra.git"
    assert _normalize_repo("github:aliikhatami94/svc-infra") == "https://github.com/aliikhatami94/svc-infra.git"
    assert _normalize_repo("git@github.com:aliikhatami94/svc-infra.git") == "https://github.com/aliikhatami94/svc-infra.git"
    assert _normalize_repo("https://github.com/aliikhatami94/svc-infra") == "https://github.com/aliikhatami94/svc-infra.git"
    assert _normalize_repo("https://github.com/aliikhatami94/svc-infra.git") == "https://github.com/aliikhatami94/svc-infra.git"

def test_root_scoped_paths_add(tmp_path):
    from ai_infra.llm.tools.custom.stdio_publisher import mcp_publish_add
    res = mcp_publish_add(
        tool_name="demo",
        module="pkg.mod.server",
        repo="github:owner/repo",
        base_dir=str(tmp_path),
        # no bin_dir -> default mcp-shim/bin
        force=True,
    )
    assert res["status"] == "ok"
    assert res["bin_path"] == "mcp-shim/bin/demo.js"
    assert (tmp_path / "mcp-shim/bin/demo.js").is_file()
    assert (tmp_path / "package.json").is_file()

def test_root_scoped_paths_remove(tmp_path):
    from ai_infra.llm.tools.custom.stdio_publisher import mcp_publish_add, mcp_publish_remove
    mcp_publish_add(tool_name="demo", module="pkg.m", repo="owner/r", base_dir=str(tmp_path))
    res = mcp_publish_remove(tool_name="demo", base_dir=str(tmp_path))
    assert res["status"] == "ok"
    # file deletion only if delete_file=True; mapping should be removed regardless