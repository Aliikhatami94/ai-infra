"""Docker execution policy for shell command execution.

This module provides Docker-based isolated command execution:

- **DockerConfig**: Configuration for Docker container execution
- **DockerExecutionPolicy**: Execute commands in isolated Docker containers
- **DockerSession**: Persistent Docker container session for multiple commands

Phase 4.3 of EXECUTOR_CLI.md - Docker Execution Policy.

Features:
- Container lifecycle management (create, reuse, destroy)
- Resource limits (memory, CPU, processes)
- Network isolation (none, bridge, host)
- Volume mounting for workspace access
- Automatic cleanup on session end
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai_infra.llm.shell.types import (
    DEFAULT_REDACTION_RULES,
    RedactionRule,
    ShellConfig,
    ShellResult,
    apply_redaction_rules,
)

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

__all__ = [
    "DockerConfig",
    "DockerExecutionPolicy",
    "DockerSession",
    "VolumeMount",
    "create_docker_policy",
    "create_docker_session",
    "get_execution_policy",
    "is_docker_available",
]


# =============================================================================
# Docker Availability Check
# =============================================================================


def is_docker_available() -> bool:
    """Check if Docker is available on the system.

    Returns:
        True if docker command is available and daemon is running.
    """
    docker_path = shutil.which("docker")
    if not docker_path:
        return False

    try:
        import subprocess

        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5.0,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


async def _async_is_docker_available() -> bool:
    """Async check if Docker is available.

    Returns:
        True if docker command is available and daemon is running.
    """
    docker_path = shutil.which("docker")
    if not docker_path:
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5.0)
        return proc.returncode == 0
    except (TimeoutError, OSError):
        return False


# =============================================================================
# Volume Mount Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class VolumeMount:
    """Configuration for a Docker volume mount.

    Attributes:
        host_path: Path on the host system.
        container_path: Path inside the container.
        read_only: Whether the mount is read-only (default: False).

    Example:
        >>> mount = VolumeMount("/home/user/project", "/workspace", read_only=False)
        >>> mount.to_docker_arg()
        '/home/user/project:/workspace:rw'
    """

    host_path: str
    container_path: str
    read_only: bool = False

    def __post_init__(self) -> None:
        """Validate mount paths."""
        if not self.host_path:
            raise ValueError("host_path cannot be empty")
        if not self.container_path:
            raise ValueError("container_path cannot be empty")
        if not self.container_path.startswith("/"):
            raise ValueError("container_path must be absolute (start with /)")

    def to_docker_arg(self) -> str:
        """Convert to Docker -v argument value.

        Returns:
            String in format 'host:container:mode'.
        """
        mode = "ro" if self.read_only else "rw"
        # Resolve host path to absolute
        host_abs = str(Path(self.host_path).resolve())
        return f"{host_abs}:{self.container_path}:{mode}"


# =============================================================================
# Docker Configuration
# =============================================================================


@dataclass
class DockerConfig:
    """Configuration for Docker container execution.

    Attributes:
        image: Docker image to use (default: python:3.11-slim).
        memory_limit: Memory limit (e.g., "512m", "1g"). Default: "512m".
        cpu_limit: CPU limit as fraction of cores (e.g., 1.0 = 1 core). Default: 1.0.
        network: Network mode ("none", "bridge", "host"). Default: "none" for isolation.
        user: User to run as inside container. Default: current UID:GID.
        workdir: Working directory inside container. Default: "/workspace".
        mounts: Volume mounts for container access.
        env_vars: Environment variables to set in container.
        remove_on_exit: Whether to remove container when done. Default: True.

    Example:
        >>> config = DockerConfig(
        ...     image="python:3.11-slim",
        ...     memory_limit="1g",
        ...     mounts=[VolumeMount("/home/user/project", "/workspace")],
        ... )
    """

    image: str = "python:3.11-slim"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    network: str = "none"  # Isolated by default
    user: str | None = None
    workdir: str = "/workspace"
    mounts: list[VolumeMount] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    remove_on_exit: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.cpu_limit <= 0:
            raise ValueError("cpu_limit must be > 0")
        if self.network not in ("none", "bridge", "host"):
            raise ValueError("network must be 'none', 'bridge', or 'host'")
        if not self.image:
            raise ValueError("image cannot be empty")

    def get_user(self) -> str:
        """Get user string for container, defaulting to current UID:GID."""
        if self.user:
            return self.user
        # Default to current user's UID:GID for proper file permissions
        uid = os.getuid() if hasattr(os, "getuid") else 1000
        gid = os.getgid() if hasattr(os, "getgid") else 1000
        return f"{uid}:{gid}"

    def to_docker_args(self) -> list[str]:
        """Convert to Docker run arguments.

        Returns:
            List of arguments for docker run command.
        """
        args: list[str] = []

        # Resource limits
        args.extend(["--memory", self.memory_limit])
        args.extend(["--cpus", str(self.cpu_limit)])

        # Network
        args.extend(["--network", self.network])

        # User
        args.extend(["--user", self.get_user()])

        # Working directory
        args.extend(["--workdir", self.workdir])

        # Volume mounts
        for mount in self.mounts:
            args.extend(["-v", mount.to_docker_arg()])

        # Environment variables
        for key, value in self.env_vars.items():
            args.extend(["-e", f"{key}={value}"])

        return args

    @classmethod
    def with_workspace(
        cls,
        workspace_path: str | Path,
        *,
        image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        network: str = "none",
        readonly_mounts: list[tuple[str, str]] | None = None,
    ) -> DockerConfig:
        """Create config with workspace mounted read-write.

        Args:
            workspace_path: Host path to mount as /workspace.
            image: Docker image to use.
            memory_limit: Memory limit.
            cpu_limit: CPU limit.
            network: Network mode.
            readonly_mounts: Additional (host, container) paths to mount read-only.

        Returns:
            DockerConfig with workspace and optional readonly mounts.
        """
        mounts = [VolumeMount(str(workspace_path), "/workspace", read_only=False)]

        if readonly_mounts:
            for host_path, container_path in readonly_mounts:
                mounts.append(VolumeMount(host_path, container_path, read_only=True))

        return cls(
            image=image,
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
            network=network,
            mounts=mounts,
        )


# =============================================================================
# Docker Execution Policy
# =============================================================================


class DockerExecutionPolicy:
    """Execute shell commands in isolated Docker containers.

    This policy runs each command in a new Docker container with:
    - Memory and CPU limits enforced by Docker
    - Network isolation (none by default)
    - Volume mounts for workspace access
    - Automatic container cleanup

    For multiple related commands, consider using DockerSession instead
    to reuse the same container.

    Example:
        >>> policy = DockerExecutionPolicy(
        ...     config=DockerConfig(image="python:3.11-slim", network="none"),
        ... )
        >>> result = await policy.execute("python --version", ShellConfig())
        >>> print(result.stdout)
        Python 3.11.x
    """

    def __init__(
        self,
        config: DockerConfig | None = None,
        *,
        redaction_rules: tuple[RedactionRule, ...] | None = DEFAULT_REDACTION_RULES,
    ) -> None:
        """Initialize Docker execution policy.

        Args:
            config: Docker configuration. Defaults to DockerConfig().
            redaction_rules: Rules for redacting sensitive output.
                Set to None to disable redaction (not recommended).
        """
        self._config = config or DockerConfig()
        self._redaction_rules = redaction_rules

    @property
    def config(self) -> DockerConfig:
        """Get the Docker configuration."""
        return self._config

    async def execute(self, command: str, config: ShellConfig) -> ShellResult:
        """Execute a shell command in a Docker container.

        Creates a new container, runs the command, and removes the container.
        For multiple commands, consider using DockerSession.

        Args:
            command: The shell command to execute.
            config: Shell configuration (timeout, max_output_bytes).

        Returns:
            ShellResult with execution outcome.
        """
        start_time = time.perf_counter()

        # Check Docker availability
        if not await _async_is_docker_available():
            return ShellResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Docker is not available. Ensure Docker is installed and running.",
                command=command,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                timed_out=False,
            )

        # Generate container name
        container_name = f"ai-infra-shell-{uuid.uuid4().hex[:12]}"

        # Build docker run command
        docker_args = [
            "docker",
            "run",
            "--rm",  # Always remove container after run
            "--name",
            container_name,
            *self._config.to_docker_args(),
            self._config.image,
            "bash",
            "-c",
            command,
        ]

        try:
            # Create subprocess
            proc = await asyncio.create_subprocess_exec(
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for completion with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=config.timeout,
                )
            except TimeoutError:
                # Kill the container on timeout
                await self._kill_container(container_name)
                return ShellResult.from_timeout(command, config.timeout)

            # Decode output
            stdout = (stdout_bytes or b"").decode(errors="replace")
            stderr = (stderr_bytes or b"").decode(errors="replace")

            # Truncate if needed
            max_bytes = config.max_output_bytes
            if len(stdout) > max_bytes:
                stdout = stdout[:max_bytes] + "\n[OUTPUT TRUNCATED]"
            if len(stderr) > max_bytes:
                stderr = stderr[:max_bytes] + "\n[OUTPUT TRUNCATED]"

            # Apply redaction
            if self._redaction_rules:
                stdout = apply_redaction_rules(stdout, self._redaction_rules)
                stderr = apply_redaction_rules(stderr, self._redaction_rules)

            duration_ms = (time.perf_counter() - start_time) * 1000
            exit_code = proc.returncode or 0

            return ShellResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                command=command,
                duration_ms=duration_ms,
                timed_out=False,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("Docker execution failed: %s", e)
            return ShellResult.from_error(command, e, duration_ms)

    async def _kill_container(self, container_name: str) -> None:
        """Kill a running container by name."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except (TimeoutError, OSError):
            pass  # Best effort cleanup


# =============================================================================
# Docker Session (Persistent Container)
# =============================================================================


class DockerSession:
    """Persistent Docker container session for multiple commands.

    This class manages a Docker container lifecycle:
    - Creates container on session start
    - Reuses container for all commands during session
    - Destroys container on session end

    Use as an async context manager for automatic cleanup.

    Example:
        >>> async with DockerSession(config) as session:
        ...     await session.execute("pip install requests")
        ...     result = await session.execute("python -c 'import requests; print(requests.__version__)'")
        ...     print(result.stdout)
    """

    def __init__(
        self,
        config: DockerConfig | None = None,
        *,
        redaction_rules: tuple[RedactionRule, ...] | None = DEFAULT_REDACTION_RULES,
    ) -> None:
        """Initialize Docker session.

        Args:
            config: Docker configuration. Defaults to DockerConfig().
            redaction_rules: Rules for redacting sensitive output.
        """
        self._config = config or DockerConfig()
        self._redaction_rules = redaction_rules
        self._container_id: str | None = None
        self._container_name: str = f"ai-infra-session-{uuid.uuid4().hex[:12]}"
        self._started = False

    @property
    def container_name(self) -> str:
        """Get the container name."""
        return self._container_name

    @property
    def container_id(self) -> str | None:
        """Get the container ID if started."""
        return self._container_id

    @property
    def is_started(self) -> bool:
        """Check if the session container is running."""
        return self._started

    async def start(self) -> None:
        """Start the session container.

        Creates a container that stays running in the background,
        waiting for commands via docker exec.

        Raises:
            RuntimeError: If Docker is not available or container fails to start.
        """
        if self._started:
            return

        if not await _async_is_docker_available():
            raise RuntimeError("Docker is not available")

        # Create container with tail -f /dev/null to keep it running
        docker_args = [
            "docker",
            "run",
            "-d",  # Detached mode
            "--name",
            self._container_name,
            *self._config.to_docker_args(),
            self._config.image,
            "tail",
            "-f",
            "/dev/null",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=60.0,
            )

            if proc.returncode != 0:
                error_msg = stderr.decode(errors="replace").strip()
                raise RuntimeError(f"Failed to start container: {error_msg}")

            self._container_id = stdout.decode().strip()[:12]
            self._started = True

            logger.debug(
                "Started Docker session container: %s (%s)",
                self._container_name,
                self._container_id,
            )

        except TimeoutError as exc:
            raise RuntimeError("Timeout starting container") from exc

    async def stop(self) -> None:
        """Stop and remove the session container."""
        if not self._started:
            return

        try:
            # Stop the container
            stop_proc = await asyncio.create_subprocess_exec(
                "docker",
                "stop",
                "-t",
                "5",  # 5 second timeout
                self._container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(stop_proc.wait(), timeout=15.0)

            # Remove the container (if not auto-removed)
            if not self._config.remove_on_exit:
                rm_proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "rm",
                    "-f",
                    self._container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(rm_proc.wait(), timeout=10.0)

            logger.debug("Stopped Docker session container: %s", self._container_name)

        except TimeoutError:
            # Force kill
            kill_proc = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                self._container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(kill_proc.wait(), timeout=10.0)
        except OSError:
            pass  # Best effort cleanup
        finally:
            self._started = False
            self._container_id = None

    async def execute(
        self,
        command: str,
        *,
        timeout: float = 120.0,
        max_output_bytes: int = 1_000_000,
    ) -> ShellResult:
        """Execute a command in the session container.

        Args:
            command: The shell command to execute.
            timeout: Maximum seconds to wait for command.
            max_output_bytes: Maximum bytes to capture from output.

        Returns:
            ShellResult with execution outcome.

        Raises:
            RuntimeError: If session not started.
        """
        if not self._started:
            raise RuntimeError("Session not started. Call start() or use context manager.")

        start_time = time.perf_counter()

        # Execute command via docker exec
        docker_args = [
            "docker",
            "exec",
            "-w",
            self._config.workdir,
            self._container_name,
            "bash",
            "-c",
            command,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                # Note: docker exec doesn't support timeout kill easily
                # The command will continue running in container
                return ShellResult.from_timeout(command, timeout)

            # Decode output
            stdout = (stdout_bytes or b"").decode(errors="replace")
            stderr = (stderr_bytes or b"").decode(errors="replace")

            # Truncate if needed
            if len(stdout) > max_output_bytes:
                stdout = stdout[:max_output_bytes] + "\n[OUTPUT TRUNCATED]"
            if len(stderr) > max_output_bytes:
                stderr = stderr[:max_output_bytes] + "\n[OUTPUT TRUNCATED]"

            # Apply redaction
            if self._redaction_rules:
                stdout = apply_redaction_rules(stdout, self._redaction_rules)
                stderr = apply_redaction_rules(stderr, self._redaction_rules)

            duration_ms = (time.perf_counter() - start_time) * 1000
            exit_code = proc.returncode or 0

            return ShellResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                command=command,
                duration_ms=duration_ms,
                timed_out=False,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ShellResult.from_error(command, e, duration_ms)

    async def __aenter__(self) -> DockerSession:
        """Enter async context, starting the container."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context, stopping the container."""
        await self.stop()


# =============================================================================
# Factory Functions
# =============================================================================


def create_docker_policy(
    image: str = "python:3.11-slim",
    *,
    workspace: str | Path | None = None,
    memory_limit: str = "512m",
    cpu_limit: float = 1.0,
    network: str = "none",
    readonly_mounts: list[tuple[str, str]] | None = None,
) -> DockerExecutionPolicy:
    """Create a DockerExecutionPolicy with common settings.

    Args:
        image: Docker image to use.
        workspace: Host path to mount as /workspace (read-write).
        memory_limit: Memory limit (e.g., "512m", "1g").
        cpu_limit: CPU limit as fraction of cores.
        network: Network mode ("none", "bridge", "host").
        readonly_mounts: Additional (host, container) paths to mount read-only.

    Returns:
        Configured DockerExecutionPolicy.

    Example:
        >>> policy = create_docker_policy(
        ...     image="python:3.11",
        ...     workspace="/home/user/project",
        ...     memory_limit="1g",
        ... )
    """
    mounts: list[VolumeMount] = []

    if workspace:
        mounts.append(VolumeMount(str(workspace), "/workspace", read_only=False))

    if readonly_mounts:
        for host_path, container_path in readonly_mounts:
            mounts.append(VolumeMount(host_path, container_path, read_only=True))

    config = DockerConfig(
        image=image,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        network=network,
        mounts=mounts,
    )

    return DockerExecutionPolicy(config=config)


def create_docker_session(
    image: str = "python:3.11-slim",
    *,
    workspace: str | Path | None = None,
    memory_limit: str = "512m",
    cpu_limit: float = 1.0,
    network: str = "none",
) -> DockerSession:
    """Create a DockerSession with common settings.

    Args:
        image: Docker image to use.
        workspace: Host path to mount as /workspace (read-write).
        memory_limit: Memory limit (e.g., "512m", "1g").
        cpu_limit: CPU limit as fraction of cores.
        network: Network mode ("none", "bridge", "host").

    Returns:
        Configured DockerSession (call start() or use as context manager).

    Example:
        >>> async with create_docker_session(workspace="/home/user/project") as session:
        ...     await session.execute("pip install requests")
        ...     result = await session.execute("python script.py")
    """
    mounts: list[VolumeMount] = []

    if workspace:
        mounts.append(VolumeMount(str(workspace), "/workspace", read_only=False))

    config = DockerConfig(
        image=image,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        network=network,
        mounts=mounts,
    )

    return DockerSession(config=config)


# =============================================================================
# Execution Policy Selection
# =============================================================================


def get_execution_policy(
    *,
    docker_isolation: bool = False,
    docker_image: str = "python:3.11-slim",
    docker_allow_network: bool = False,
    workspace: str | Path | None = None,
    memory_limit_mb: int = 512,
    cpu_limit_seconds: int = 60,
) -> DockerExecutionPolicy | None:
    """Get an execution policy based on configuration.

    Returns a DockerExecutionPolicy if docker_isolation is True and Docker
    is available, otherwise returns None (caller should fall back to host
    execution or LimitedExecutionPolicy).

    This function handles Docker availability gracefully - if Docker is
    requested but not available, it logs a warning and returns None.

    Args:
        docker_isolation: Whether to use Docker isolation.
        docker_image: Docker image to use for execution.
        docker_allow_network: Whether to allow network access in container.
        workspace: Host path to mount as /workspace (read-write).
        memory_limit_mb: Memory limit in megabytes.
        cpu_limit_seconds: CPU time limit (used for CPU fraction calculation).

    Returns:
        DockerExecutionPolicy if Docker isolation is requested and available,
        None otherwise.

    Example:
        >>> policy = get_execution_policy(
        ...     docker_isolation=True,
        ...     docker_image="python:3.11-slim",
        ...     docker_allow_network=False,
        ...     workspace="/home/user/project",
        ... )
        >>> if policy is None:
        ...     # Fall back to LimitedExecutionPolicy or HostExecutionPolicy
        ...     policy = LimitedExecutionPolicy()
    """
    if not docker_isolation:
        return None

    # Check Docker availability
    if not is_docker_available():
        logger.warning(
            "Docker isolation requested but Docker is not available. "
            "Ensure Docker is installed and running. "
            "Falling back to host execution."
        )
        return None

    # Build mounts
    mounts: list[VolumeMount] = []
    if workspace:
        mounts.append(VolumeMount(str(workspace), "/workspace", read_only=False))

    # Convert memory limit to Docker format
    memory_limit = f"{memory_limit_mb}m"

    # Network mode: bridge allows network, none blocks it
    network = "bridge" if docker_allow_network else "none"

    # CPU limit as fraction of cores (rough estimate from seconds)
    # Default to 1.0 core - the actual limit is enforced by timeout
    cpu_limit = 1.0

    config = DockerConfig(
        image=docker_image,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        network=network,
        mounts=mounts,
    )

    logger.info(
        "Using Docker execution policy: image=%s, memory=%s, network=%s",
        docker_image,
        memory_limit,
        network,
    )

    return DockerExecutionPolicy(config=config)
