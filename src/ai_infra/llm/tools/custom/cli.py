import sys
import asyncio

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
            "-ExecutionPolicy", "Bypass",
            "-Command", command,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-lc", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    stdout_b, stderr_b = await proc.communicate()
    code = proc.returncode
    out = (stdout_b or b"").decode(errors="replace")
    err = (stderr_b or b"").decode(errors="replace")

    if code != 0:
        raise RuntimeError(
            f"Command failed with code {code}\n"
            f"STDOUT:\n{out}\n"
            f"STDERR:\n{err}"
        )
    return out.strip()
