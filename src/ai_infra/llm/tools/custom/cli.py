import asyncio
import sys

from svc_infra.app.env import prepare_env
from svc_infra.cli.foundation.runner import run_from_root


async def run_cli(command: str) -> str:
    """
    Run a shell command asynchronously and return its stdout as a string.
    - Windows: PowerShell for better pipelines/globbing
    - Unix: bash -lc for predictable shell semantics
    Raises RuntimeError on non-zero exit with stdout/stderr attached.
    """
    if sys.platform.startswith("win"):
        args = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    stdout_b, stderr_b = await proc.communicate()
    code = proc.returncode
    out = (stdout_b or b"").decode(errors="replace")
    err = (stderr_b or b"").decode(errors="replace")

    if code != 0:
        raise RuntimeError(f"Command failed with code {code}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out.strip()


async def cli_cmd_help(cli_prog: str) -> dict:
    root = prepare_env()
    text = await run_from_root(root, cli_prog, ["--help"])
    return {"ok": True, "action": "help", "project_root": str(root), "help": text}


async def cli_subcmd_help(cli_prog: str, subcommand) -> dict:
    root = prepare_env()
    cmd = subcommand.value
    text = await run_from_root(root, cli_prog, [cmd, "--help"])
    return {
        "ok": True,
        "action": "subcommand_help",
        "subcommand": cmd,
        "project_root": str(root),
        "help": text,
    }
