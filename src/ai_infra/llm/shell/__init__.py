"""Shell execution module for ai-infra.

This module provides production-grade shell command execution capabilities:

- **ShellResult**: Result of a shell command execution
- **ShellConfig**: Configuration for shell execution
- **ExecutionPolicy**: Protocol for execution strategies
- **HostExecutionPolicy**: Direct host execution (default)
- **DockerExecutionPolicy**: Isolated Docker container execution (Phase 4.3)
- **RedactionRule**: Pattern-based output sanitization
- **ShellSession**: Persistent shell session with state preservation
- **SessionConfig**: Configuration for shell sessions

Example (one-shot execution):
    >>> from ai_infra.llm.shell import HostExecutionPolicy, ShellConfig
    >>>
    >>> policy = HostExecutionPolicy()
    >>> config = ShellConfig(timeout=30.0)
    >>> result = await policy.execute("echo hello", config)
    >>> print(result.stdout)
    hello

Example (persistent session):
    >>> from ai_infra.llm.shell import ShellSession
    >>>
    >>> async with ShellSession() as session:
    ...     await session.execute("export FOO=bar")
    ...     result = await session.execute("echo $FOO")
    ...     print(result.stdout)  # bar

Example (Docker isolated execution):
    >>> from ai_infra.llm.shell import DockerExecutionPolicy, DockerConfig
    >>>
    >>> config = DockerConfig(image="python:3.11-slim", network="none")
    >>> policy = DockerExecutionPolicy(config=config)
    >>> result = await policy.execute("python --version", ShellConfig())

Phase 1 of EXECUTOR_CLI.md - Shell Tool Integration.
"""

from ai_infra.llm.shell.audit import (
    AuditEvent,
    AuditEventType,
    AuditReport,
    RedactionEvent,
    SecurityViolationEvent,
    ShellAuditLogger,
    generate_audit_report,
    get_shell_audit_logger,
    set_shell_audit_logger,
)
from ai_infra.llm.shell.docker import (
    DockerConfig,
    DockerExecutionPolicy,
    DockerSession,
    VolumeMount,
    create_docker_policy,
    create_docker_session,
    get_execution_policy,
    is_docker_available,
)
from ai_infra.llm.shell.helpers import (
    check_command_exists,
    cli_cmd_help,
    cli_help,
    cli_subcmd_help,
    run_command,
    run_command_sync,
)
from ai_infra.llm.shell.limits import (
    DEFAULT_RESOURCE_LIMITS,
    LimitedExecutionPolicy,
    ResourceLimits,
    create_limit_prelude,
    create_limited_policy,
    is_limits_supported,
)
from ai_infra.llm.shell.middleware import ShellMiddleware, ShellMiddlewareConfig
from ai_infra.llm.shell.security import (
    DEFAULT_ALLOWED_PATTERNS,
    DEFAULT_DENIED_PATTERNS,
    SecurityPolicy,
    ValidationResult,
    ValidationStatus,
    create_permissive_policy,
    create_strict_policy,
    is_network_command,
    validate_command,
    validate_command_with_network,
)
from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.tool import (
    DANGEROUS_PATTERNS,
    create_shell_tool,
    get_current_session,
    run_shell,
    set_current_session,
)
from ai_infra.llm.shell.types import (
    DEFAULT_REDACTION_RULES,
    ExecutionPolicy,
    HostExecutionPolicy,
    RedactionRule,
    ShellConfig,
    ShellResult,
    apply_redaction_rules,
)

__all__ = [
    # Core types
    "ShellResult",
    "ShellConfig",
    # Execution policies
    "ExecutionPolicy",
    "HostExecutionPolicy",
    "LimitedExecutionPolicy",
    # Resource limits (Phase 4.2)
    "ResourceLimits",
    "DEFAULT_RESOURCE_LIMITS",
    "create_limited_policy",
    "create_limit_prelude",
    "is_limits_supported",
    # Session management
    "ShellSession",
    "SessionConfig",
    # Redaction
    "RedactionRule",
    "DEFAULT_REDACTION_RULES",
    "apply_redaction_rules",
    # Security (Phase 4.1)
    "SecurityPolicy",
    "ValidationResult",
    "ValidationStatus",
    "validate_command",
    "validate_command_with_network",
    "is_network_command",
    "create_strict_policy",
    "create_permissive_policy",
    "DEFAULT_ALLOWED_PATTERNS",
    "DEFAULT_DENIED_PATTERNS",
    # Tool
    "run_shell",
    "create_shell_tool",
    "get_current_session",
    "set_current_session",
    "DANGEROUS_PATTERNS",
    # Middleware
    "ShellMiddleware",
    "ShellMiddlewareConfig",
    # Helpers
    "check_command_exists",
    "cli_cmd_help",
    "cli_help",
    "cli_subcmd_help",
    "run_command",
    "run_command_sync",
    # Docker execution (Phase 4.3)
    "DockerConfig",
    "DockerExecutionPolicy",
    "DockerSession",
    "VolumeMount",
    "create_docker_policy",
    "create_docker_session",
    "get_execution_policy",
    "is_docker_available",
    # Audit logging (Phase 4.4)
    "ShellAuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditReport",
    "RedactionEvent",
    "SecurityViolationEvent",
    "generate_audit_report",
    "get_shell_audit_logger",
    "set_shell_audit_logger",
]
