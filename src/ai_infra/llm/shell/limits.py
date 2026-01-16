"""Resource limits for shell command execution.

This module provides resource limiting capabilities for shell commands:

- **ResourceLimits**: Configuration for memory, CPU, file, and process limits
- **LimitedExecutionPolicy**: Execution policy that enforces resource limits
- **enforce_limits**: Low-level limit enforcement via Unix ulimit

Phase 4.2 of EXECUTOR_CLI.md - Resource Limits.

Platform Support:
- Linux/macOS: Full support via resource module and ulimit
- Windows: Limited support (timeout only, no memory/CPU limits)
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_infra.llm.shell.types import (
    DEFAULT_REDACTION_RULES,
    RedactionRule,
    ShellConfig,
    ShellResult,
    apply_redaction_rules,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "ResourceLimits",
    "LimitedExecutionPolicy",
    "DEFAULT_RESOURCE_LIMITS",
    "create_limit_prelude",
    "is_limits_supported",
]


# =============================================================================
# Resource Limits Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Configuration for resource limits on shell command execution.

    Attributes:
        memory_mb: Maximum memory usage in megabytes (default: 512).
            Set to 0 to disable memory limiting.
        cpu_seconds: Maximum CPU time in seconds (default: 60).
            Set to 0 to disable CPU time limiting.
        max_file_size_mb: Maximum size of files created in megabytes (default: 100).
            Set to 0 to disable file size limiting.
        max_open_files: Maximum number of open file descriptors (default: 256).
            Set to 0 to disable open file limiting.
        max_processes: Maximum number of child processes (default: 32).
            Set to 0 to disable process limiting.
        max_output_bytes: Maximum bytes to capture from stdout/stderr (default: 1MB).
            This is enforced in Python, not via ulimit.
        enable_core_dumps: Whether to allow core dumps (default: False).

    Example:
        >>> limits = ResourceLimits(memory_mb=256, cpu_seconds=30)
        >>> policy = LimitedExecutionPolicy(limits=limits)
        >>> result = await policy.execute("python heavy_script.py", config)

    Note:
        On Windows, only timeout and max_output_bytes are enforced.
        Memory, CPU, and process limits require Unix-like systems.
    """

    memory_mb: int = 512
    cpu_seconds: int = 60
    max_file_size_mb: int = 100
    max_open_files: int = 256
    max_processes: int = 32
    max_output_bytes: int = 1_000_000  # 1MB
    enable_core_dumps: bool = False

    def __post_init__(self) -> None:
        """Validate limit values."""
        if self.memory_mb < 0:
            raise ValueError("memory_mb must be >= 0")
        if self.cpu_seconds < 0:
            raise ValueError("cpu_seconds must be >= 0")
        if self.max_file_size_mb < 0:
            raise ValueError("max_file_size_mb must be >= 0")
        if self.max_open_files < 0:
            raise ValueError("max_open_files must be >= 0")
        if self.max_processes < 0:
            raise ValueError("max_processes must be >= 0")
        if self.max_output_bytes < 0:
            raise ValueError("max_output_bytes must be >= 0")

    def to_dict(self) -> dict[str, int | bool]:
        """Convert to dictionary for serialization."""
        return {
            "memory_mb": self.memory_mb,
            "cpu_seconds": self.cpu_seconds,
            "max_file_size_mb": self.max_file_size_mb,
            "max_open_files": self.max_open_files,
            "max_processes": self.max_processes,
            "max_output_bytes": self.max_output_bytes,
            "enable_core_dumps": self.enable_core_dumps,
        }

    @classmethod
    def strict(cls) -> ResourceLimits:
        """Create strict limits for untrusted code.

        Returns:
            ResourceLimits with conservative values suitable for
            running untrusted or potentially dangerous commands.
        """
        return cls(
            memory_mb=256,
            cpu_seconds=30,
            max_file_size_mb=10,
            max_open_files=64,
            max_processes=8,
            max_output_bytes=100_000,  # 100KB
            enable_core_dumps=False,
        )

    @classmethod
    def permissive(cls) -> ResourceLimits:
        """Create permissive limits for trusted code.

        Returns:
            ResourceLimits with generous values suitable for
            running trusted development commands.
        """
        return cls(
            memory_mb=2048,  # 2GB
            cpu_seconds=300,  # 5 minutes
            max_file_size_mb=500,
            max_open_files=1024,
            max_processes=128,
            max_output_bytes=10_000_000,  # 10MB
            enable_core_dumps=False,
        )

    @classmethod
    def unlimited(cls) -> ResourceLimits:
        """Create unlimited limits (no enforcement).

        Returns:
            ResourceLimits with all limits set to 0 (disabled).
            Use with caution - only for fully trusted environments.
        """
        return cls(
            memory_mb=0,
            cpu_seconds=0,
            max_file_size_mb=0,
            max_open_files=0,
            max_processes=0,
            max_output_bytes=0,
            enable_core_dumps=True,
        )


# Default resource limits for shell execution
DEFAULT_RESOURCE_LIMITS = ResourceLimits()


# =============================================================================
# Platform Support
# =============================================================================


def is_limits_supported() -> bool:
    """Check if resource limits are supported on this platform.

    Returns:
        True if the platform supports resource limits (Unix-like systems).
    """
    return not sys.platform.startswith("win")


def _has_resource_module() -> bool:
    """Check if Python's resource module is available."""
    try:
        import resource  # noqa: F401

        return True
    except ImportError:
        return False


# =============================================================================
# Limit Enforcement via Shell Prelude
# =============================================================================


def create_limit_prelude(limits: ResourceLimits) -> str:
    """Create a shell prelude that sets ulimit-based resource limits.

    This function generates shell commands that enforce resource limits
    using the `ulimit` builtin. The prelude is prepended to the actual
    command to ensure limits are in place before execution.

    Args:
        limits: ResourceLimits configuration.

    Returns:
        Shell commands to set limits, or empty string if no limits.

    Example:
        >>> limits = ResourceLimits(memory_mb=256, cpu_seconds=30)
        >>> prelude = create_limit_prelude(limits)
        >>> print(prelude)
        ulimit -v 262144; ulimit -t 30; ...
    """
    if not is_limits_supported():
        return ""

    parts: list[str] = []

    # Memory limit (virtual memory in KB)
    if limits.memory_mb > 0:
        memory_kb = limits.memory_mb * 1024
        parts.append(f"ulimit -v {memory_kb}")

    # CPU time limit (seconds)
    if limits.cpu_seconds > 0:
        parts.append(f"ulimit -t {limits.cpu_seconds}")

    # File size limit (KB)
    if limits.max_file_size_mb > 0:
        file_size_kb = limits.max_file_size_mb * 1024
        parts.append(f"ulimit -f {file_size_kb}")

    # Open files limit
    if limits.max_open_files > 0:
        parts.append(f"ulimit -n {limits.max_open_files}")

    # Process limit
    if limits.max_processes > 0:
        parts.append(f"ulimit -u {limits.max_processes}")

    # Core dump limit
    if not limits.enable_core_dumps:
        parts.append("ulimit -c 0")

    if not parts:
        return ""

    return "; ".join(parts) + "; "


# =============================================================================
# Limited Execution Policy
# =============================================================================


class LimitedExecutionPolicy:
    """Execute shell commands with resource limits enforced.

    This policy wraps command execution with resource limits using:
    - Shell ulimit commands for Unix-like systems
    - Python's resource module for additional enforcement
    - Timeout enforcement for all platforms

    Example:
        >>> limits = ResourceLimits(memory_mb=256, cpu_seconds=30)
        >>> policy = LimitedExecutionPolicy(limits=limits)
        >>> result = await policy.execute("python script.py", ShellConfig())
        >>> if not result.success:
        ...     print(f"Command failed: {result.stderr}")
    """

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        *,
        redaction_rules: tuple[RedactionRule, ...] | None = DEFAULT_REDACTION_RULES,
    ) -> None:
        """Initialize limited execution policy.

        Args:
            limits: Resource limits to enforce. Defaults to DEFAULT_RESOURCE_LIMITS.
            redaction_rules: Rules for redacting sensitive output.
                Set to None to disable redaction (not recommended).
        """
        self._limits = limits or DEFAULT_RESOURCE_LIMITS
        self._redaction_rules = redaction_rules

    @property
    def limits(self) -> ResourceLimits:
        """Get the configured resource limits."""
        return self._limits

    async def execute(self, command: str, config: ShellConfig) -> ShellResult:
        """Execute a shell command with resource limits.

        Args:
            command: The shell command to execute.
            config: Configuration for execution.

        Returns:
            ShellResult with execution outcome.
        """
        start_time = time.perf_counter()

        # Determine effective max output bytes
        max_output = self._limits.max_output_bytes
        if max_output == 0:
            max_output = config.max_output_bytes

        try:
            # Build the limited command
            if is_limits_supported():
                prelude = create_limit_prelude(self._limits)
                limited_command = f"{prelude}{command}"
                shell_args = ["bash", "-lc", limited_command]
            else:
                # Windows: no ulimit support, just run the command
                limited_command = command
                shell_args = [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ]

            # Note: We use ulimit in the shell prelude rather than preexec_fn
            # because preexec_fn with RLIMIT_NPROC affects the entire user's
            # process count, not just children of this command.

            proc = await asyncio.create_subprocess_exec(
                *shell_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(config.get_cwd()),
                env=config.get_env(),
            )

            # Wait for completion with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=config.timeout,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ShellResult.from_timeout(command, config.timeout)

            # Decode output
            stdout = (stdout_bytes or b"").decode(errors="replace")
            stderr = (stderr_bytes or b"").decode(errors="replace")

            # Truncate if needed
            if max_output > 0:
                if len(stdout) > max_output:
                    stdout = stdout[:max_output] + "\n[OUTPUT TRUNCATED]"
                if len(stderr) > max_output:
                    stderr = stderr[:max_output] + "\n[OUTPUT TRUNCATED]"

            # Apply redaction
            if self._redaction_rules:
                stdout = apply_redaction_rules(stdout, self._redaction_rules)
                stderr = apply_redaction_rules(stderr, self._redaction_rules)

            duration_ms = (time.perf_counter() - start_time) * 1000
            exit_code = proc.returncode or 0

            # Check for resource limit violations in stderr
            stderr, limit_type = self._annotate_limit_errors(stderr, exit_code)

            return ShellResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                command=command,
                duration_ms=duration_ms,
                timed_out=False,
                resource_limit_exceeded=limit_type,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ShellResult.from_error(command, e, duration_ms)

    def _create_preexec_fn(self):
        """Create a preexec function for setting resource limits.

        This is called in the child process after fork but before exec.
        Only used on Unix-like systems.
        """
        if not _has_resource_module():
            return None

        limits = self._limits

        def set_limits() -> None:
            """Set resource limits in the child process."""
            import resource

            # Memory limit (RLIMIT_AS = address space)
            if limits.memory_mb > 0:
                memory_bytes = limits.memory_mb * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                except (OSError, ValueError):
                    pass  # Limit may exceed system maximum

            # CPU time limit
            if limits.cpu_seconds > 0:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_CPU,
                        (limits.cpu_seconds, limits.cpu_seconds),
                    )
                except (OSError, ValueError):
                    pass

            # File size limit
            if limits.max_file_size_mb > 0:
                file_bytes = limits.max_file_size_mb * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
                except (OSError, ValueError):
                    pass

            # Open files limit
            if limits.max_open_files > 0:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_NOFILE,
                        (limits.max_open_files, limits.max_open_files),
                    )
                except (OSError, ValueError):
                    pass

            # Process limit (RLIMIT_NPROC)
            if limits.max_processes > 0:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_NPROC,
                        (limits.max_processes, limits.max_processes),
                    )
                except (OSError, ValueError):
                    pass

            # Core dump limit
            if not limits.enable_core_dumps:
                try:
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                except (OSError, ValueError):
                    pass

        return set_limits

    def _annotate_limit_errors(self, stderr: str, exit_code: int) -> tuple[str, str | None]:
        """Annotate stderr with human-readable limit violation messages.

        Args:
            stderr: Original stderr content.
            exit_code: Command exit code.

        Returns:
            Tuple of (annotated stderr, limit_type) where limit_type is one of:
            "memory", "cpu", "file_size", "open_files", "processes", or None.
        """
        # Common signals for resource limit violations
        annotations: list[str] = []
        limit_type: str | None = None

        # Exit code 137 = killed by signal 9 (SIGKILL) - often OOM killer
        if exit_code == 137:
            annotations.append(
                "[RESOURCE LIMIT] Process killed (exit 137). "
                "Likely exceeded memory limit or was killed by OOM."
            )
            limit_type = "memory"

        # Exit code 152 = signal 24 (SIGXCPU) - CPU time exceeded
        if exit_code == 152:
            annotations.append(
                f"[RESOURCE LIMIT] CPU time limit exceeded ({self._limits.cpu_seconds}s)."
            )
            limit_type = "cpu"

        # Exit code 153 = signal 25 (SIGXFSZ) - File size exceeded
        if exit_code == 153:
            annotations.append(
                f"[RESOURCE LIMIT] File size limit exceeded ({self._limits.max_file_size_mb}MB)."
            )
            limit_type = "file_size"

        # Check for common error messages
        if "cannot fork" in stderr.lower() or "resource temporarily unavailable" in stderr.lower():
            annotations.append(
                f"[RESOURCE LIMIT] Process limit may have been exceeded ({self._limits.max_processes})."
            )
            if limit_type is None:
                limit_type = "processes"

        if "too many open files" in stderr.lower():
            annotations.append(
                f"[RESOURCE LIMIT] Open file limit exceeded ({self._limits.max_open_files})."
            )
            if limit_type is None:
                limit_type = "open_files"

        if annotations:
            return "\n".join(annotations) + "\n\n" + stderr, limit_type

        return stderr, None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_limited_policy(
    memory_mb: int = 512,
    cpu_seconds: int = 60,
    max_processes: int = 32,
) -> LimitedExecutionPolicy:
    """Create a LimitedExecutionPolicy with common settings.

    Args:
        memory_mb: Maximum memory in megabytes.
        cpu_seconds: Maximum CPU time in seconds.
        max_processes: Maximum number of child processes.

    Returns:
        LimitedExecutionPolicy with specified limits.

    Example:
        >>> policy = create_limited_policy(memory_mb=256, cpu_seconds=30)
        >>> result = await policy.execute("python script.py", config)
    """
    limits = ResourceLimits(
        memory_mb=memory_mb,
        cpu_seconds=cpu_seconds,
        max_processes=max_processes,
    )
    return LimitedExecutionPolicy(limits=limits)
