import sys
import subprocess

from langchain_core.tools import tool

@tool
def run_command(command: str) -> str:
    """
    Run a shell command and return its output as a string.
    Uses bash on Unix and PowerShell on Windows for better parity.
    """
    if sys.platform.startswith("win"):
        # PowerShell: good for pipelines/globbing, avoids cmd quirks
        ps_cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy", "Bypass",
            "-Command", command,
        ]
        result = subprocess.run(ps_cmd, text=True, capture_output=True)
    else:
        # Use bash -lc so things like $PWD, pipes, &&, set -e behave predictably
        result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return (result.stdout or "").strip()