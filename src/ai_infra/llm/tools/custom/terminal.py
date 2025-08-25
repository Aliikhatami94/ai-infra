import subprocess

from langchain_core.tools import tool

@tool
def run_command(command: str) -> str:
    """
    Run a shell command and return its output as a string.
    Raises an exception if the command fails.
    """
    result = subprocess.run(
        command,
        shell=True,          # run in the shell
        text=True,           # decode bytes to str
        capture_output=True  # capture stdout + stderr
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result.stdout.strip()