"""Audit logging for shell command execution.

This module provides structured audit logging for shell operations:

- **ShellAuditLogger**: Logger for shell command audit events
- **AuditEvent**: Structured audit event data
- **RedactionEvent**: Event for detected secrets redaction
- **SecurityViolationEvent**: Event for security policy violations

Phase 4.4 of EXECUTOR_CLI.md - Audit Logging.

Usage:
    from ai_infra.llm.shell.audit import get_shell_audit_logger

    audit = get_shell_audit_logger()

    # Log command execution
    audit.log_command(command="echo hello", exit_code=0, duration_ms=5.2)

    # Log redaction
    audit.log_redaction(rule_name="openai_api_key", count=1)

    # Log security violation
    audit.log_security_violation(command="rm -rf /", reason="Denied pattern match")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.llm.shell.security import ValidationResult
    from ai_infra.llm.shell.types import ShellResult

__all__ = [
    "ShellAuditLogger",
    "AuditEvent",
    "AuditEventType",
    "RedactionEvent",
    "SecurityViolationEvent",
    "get_shell_audit_logger",
    "set_shell_audit_logger",
]


# =============================================================================
# Audit Event Types
# =============================================================================


class AuditEventType(str, Enum):
    """Types of audit events."""

    COMMAND_EXECUTED = "command_executed"
    COMMAND_FAILED = "command_failed"
    COMMAND_TIMEOUT = "command_timeout"
    SECRET_REDACTED = "secret_redacted"
    SECURITY_VIOLATION = "security_violation"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"


# =============================================================================
# Audit Event Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Base audit event for shell operations.

    Attributes:
        event_type: Type of audit event.
        timestamp: ISO 8601 timestamp when event occurred.
        command: The shell command (may be truncated for security).
        exit_code: Command exit code (-1 for errors/timeout).
        duration_ms: Execution time in milliseconds.
        success: Whether command succeeded.
        session_id: Optional session identifier.
        container_id: Optional Docker container ID.
        user: User who executed the command.
        extra: Additional event-specific data.
    """

    event_type: AuditEventType
    timestamp: str
    command: str
    exit_code: int = 0
    duration_ms: float = 0.0
    success: bool = True
    session_id: str | None = None
    container_id: str | None = None
    user: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        d: dict[str, Any] = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "command": self.command,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
        }
        if self.session_id:
            d["session_id"] = self.session_id
        if self.container_id:
            d["container_id"] = self.container_id
        if self.user:
            d["user"] = self.user
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_result(
        cls,
        result: ShellResult,
        *,
        session_id: str | None = None,
        container_id: str | None = None,
        user: str | None = None,
    ) -> AuditEvent:
        """Create audit event from ShellResult.

        Args:
            result: Shell execution result.
            session_id: Optional session identifier.
            container_id: Optional Docker container ID.
            user: User who executed the command.

        Returns:
            AuditEvent with appropriate type based on result.
        """
        if result.timed_out:
            event_type = AuditEventType.COMMAND_TIMEOUT
        elif result.success:
            event_type = AuditEventType.COMMAND_EXECUTED
        else:
            event_type = AuditEventType.COMMAND_FAILED

        return cls(
            event_type=event_type,
            timestamp=datetime.now(UTC).isoformat(),
            command=_truncate_command(result.command),
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            success=result.success,
            session_id=session_id,
            container_id=container_id,
            user=user,
        )


@dataclass(frozen=True, slots=True)
class RedactionEvent:
    """Audit event for secret redaction.

    Attributes:
        timestamp: ISO 8601 timestamp.
        rule_name: Name of the redaction rule that matched.
        count: Number of matches redacted.
        command: The command where secrets were found (truncated).
        session_id: Optional session identifier.
    """

    timestamp: str
    rule_name: str
    count: int
    command: str
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        d: dict[str, Any] = {
            "event_type": AuditEventType.SECRET_REDACTED.value,
            "timestamp": self.timestamp,
            "rule_name": self.rule_name,
            "count": self.count,
            "command": self.command,
        }
        if self.session_id:
            d["session_id"] = self.session_id
        return d


@dataclass(frozen=True, slots=True)
class SecurityViolationEvent:
    """Audit event for security policy violations.

    Attributes:
        timestamp: ISO 8601 timestamp.
        command: The command that violated policy (truncated).
        reason: Description of why command was denied.
        policy_name: Name of the policy that was violated.
        matched_pattern: The pattern that matched (if applicable).
        session_id: Optional session identifier.
        user: User who attempted the command.
    """

    timestamp: str
    command: str
    reason: str
    policy_name: str = "default"
    matched_pattern: str | None = None
    session_id: str | None = None
    user: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        d: dict[str, Any] = {
            "event_type": AuditEventType.SECURITY_VIOLATION.value,
            "timestamp": self.timestamp,
            "command": self.command,
            "reason": self.reason,
            "policy_name": self.policy_name,
        }
        if self.matched_pattern:
            d["matched_pattern"] = self.matched_pattern
        if self.session_id:
            d["session_id"] = self.session_id
        if self.user:
            d["user"] = self.user
        return d

    @classmethod
    def from_validation_result(
        cls,
        command: str,
        result: ValidationResult,
        *,
        policy_name: str = "default",
        session_id: str | None = None,
        user: str | None = None,
    ) -> SecurityViolationEvent:
        """Create from a ValidationResult.

        Args:
            command: The command that was validated.
            result: ValidationResult from security check.
            policy_name: Name of the security policy.
            session_id: Optional session identifier.
            user: User who attempted the command.

        Returns:
            SecurityViolationEvent with details from validation result.
        """
        return cls(
            timestamp=datetime.now(UTC).isoformat(),
            command=_truncate_command(command),
            reason=result.reason or "Security policy violation",
            policy_name=policy_name,
            matched_pattern=result.matched_pattern,
            session_id=session_id,
            user=user,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _truncate_command(command: str, max_length: int = 200) -> str:
    """Truncate command for safe logging.

    Args:
        command: The command to truncate.
        max_length: Maximum length to keep.

    Returns:
        Truncated command with ellipsis if needed.
    """
    if len(command) <= max_length:
        return command
    return command[:max_length] + "..."


def _get_current_user() -> str | None:
    """Get current user for audit logging."""
    import os

    return os.environ.get("USER") or os.environ.get("USERNAME")


# =============================================================================
# Shell Audit Logger
# =============================================================================


class ShellAuditLogger:
    """Structured audit logger for shell operations.

    Provides methods for logging shell-related audit events with
    consistent structure and fields.

    Example:
        audit = ShellAuditLogger()

        # Log command execution
        audit.log_command(command="echo hello", exit_code=0, duration_ms=5.2)

        # Log from ShellResult
        audit.log_result(result)

        # Log security violation
        audit.log_security_violation(
            command="rm -rf /",
            reason="Matches denied pattern",
            matched_pattern="rm\\s+-rf",
        )
    """

    def __init__(
        self,
        name: str = "shell.audit",
        *,
        enabled: bool = True,
        include_user: bool = True,
    ) -> None:
        """Initialize audit logger.

        Args:
            name: Logger name (prefixed with ai_infra.).
            enabled: Whether audit logging is enabled.
            include_user: Whether to include current user in logs.
        """
        self._logger = logging.getLogger(f"ai_infra.{name}")
        self._enabled = enabled
        self._include_user = include_user

    @property
    def enabled(self) -> bool:
        """Whether audit logging is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable audit logging."""
        self._enabled = value

    def _log(self, level: int, message: str, **fields: Any) -> None:
        """Internal logging method."""
        if self._enabled and self._logger.isEnabledFor(level):
            self._logger.log(level, message, extra=fields)

    def _get_user(self) -> str | None:
        """Get user for audit log if enabled."""
        return _get_current_user() if self._include_user else None

    # -------------------------------------------------------------------------
    # Command Execution Logging
    # -------------------------------------------------------------------------

    def log_command(
        self,
        command: str,
        exit_code: int,
        duration_ms: float,
        *,
        success: bool | None = None,
        timed_out: bool = False,
        session_id: str | None = None,
        container_id: str | None = None,
    ) -> None:
        """Log a shell command execution.

        Args:
            command: The command that was executed.
            exit_code: Command exit code.
            duration_ms: Execution time in milliseconds.
            success: Whether command succeeded (defaults to exit_code == 0).
            timed_out: Whether command timed out.
            session_id: Optional session identifier.
            container_id: Optional Docker container ID.
        """
        if success is None:
            success = exit_code == 0

        # Determine event type
        if timed_out:
            event_type = AuditEventType.COMMAND_TIMEOUT
            level = logging.WARNING
        elif success:
            event_type = AuditEventType.COMMAND_EXECUTED
            level = logging.INFO
        else:
            event_type = AuditEventType.COMMAND_FAILED
            level = logging.WARNING

        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC).isoformat(),
            command=_truncate_command(command),
            exit_code=exit_code,
            duration_ms=duration_ms,
            success=success,
            session_id=session_id,
            container_id=container_id,
            user=self._get_user(),
        )

        self._log(level, "shell_command", **event.to_dict())

    def log_result(
        self,
        result: ShellResult,
        *,
        session_id: str | None = None,
        container_id: str | None = None,
    ) -> None:
        """Log from a ShellResult object.

        Args:
            result: Shell execution result.
            session_id: Optional session identifier.
            container_id: Optional Docker container ID.
        """
        self.log_command(
            command=result.command,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            success=result.success,
            timed_out=result.timed_out,
            session_id=session_id,
            container_id=container_id,
        )

    # -------------------------------------------------------------------------
    # Redaction Logging
    # -------------------------------------------------------------------------

    def log_redaction(
        self,
        rule_name: str,
        count: int = 1,
        *,
        command: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log that secrets were redacted from output.

        Args:
            rule_name: Name of the redaction rule that matched.
            count: Number of matches redacted.
            command: The command where secrets were found.
            session_id: Optional session identifier.
        """
        event = RedactionEvent(
            timestamp=datetime.now(UTC).isoformat(),
            rule_name=rule_name,
            count=count,
            command=_truncate_command(command or ""),
            session_id=session_id,
        )

        self._log(logging.WARNING, "secret_redacted", **event.to_dict())

    def log_redactions(
        self,
        redactions: dict[str, int],
        *,
        command: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log multiple redaction events.

        Args:
            redactions: Mapping of rule_name -> count.
            command: The command where secrets were found.
            session_id: Optional session identifier.
        """
        for rule_name, count in redactions.items():
            if count > 0:
                self.log_redaction(
                    rule_name=rule_name,
                    count=count,
                    command=command,
                    session_id=session_id,
                )

    # -------------------------------------------------------------------------
    # Security Violation Logging
    # -------------------------------------------------------------------------

    def log_security_violation(
        self,
        command: str,
        reason: str,
        *,
        policy_name: str = "default",
        matched_pattern: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log a security policy violation.

        Args:
            command: The command that violated policy.
            reason: Description of why command was denied.
            policy_name: Name of the security policy.
            matched_pattern: The pattern that matched (if applicable).
            session_id: Optional session identifier.
        """
        event = SecurityViolationEvent(
            timestamp=datetime.now(UTC).isoformat(),
            command=_truncate_command(command),
            reason=reason,
            policy_name=policy_name,
            matched_pattern=matched_pattern,
            session_id=session_id,
            user=self._get_user(),
        )

        self._log(logging.WARNING, "security_violation", **event.to_dict())

    def log_validation_result(
        self,
        command: str,
        result: ValidationResult,
        *,
        policy_name: str = "default",
        session_id: str | None = None,
    ) -> None:
        """Log a security validation result if it's a violation.

        Args:
            command: The command that was validated.
            result: ValidationResult from security check.
            policy_name: Name of the security policy.
            session_id: Optional session identifier.
        """
        # Only log violations (denied commands)
        from ai_infra.llm.shell.security import ValidationStatus

        if result.status != ValidationStatus.ALLOWED:
            self.log_security_violation(
                command=command,
                reason=result.reason or "Security policy violation",
                policy_name=policy_name,
                matched_pattern=result.matched_pattern,
                session_id=session_id,
            )

    # -------------------------------------------------------------------------
    # Session Logging
    # -------------------------------------------------------------------------

    def log_session_started(
        self,
        session_id: str,
        *,
        container_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Log session start event.

        Args:
            session_id: Session identifier.
            container_id: Docker container ID if applicable.
            config: Session configuration details.
        """
        extra: dict[str, Any] = {
            "event_type": AuditEventType.SESSION_STARTED.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session_id,
        }
        if container_id:
            extra["container_id"] = container_id
        if config:
            extra["config"] = config
        if user := self._get_user():
            extra["user"] = user

        self._log(logging.INFO, "session_started", **extra)

    def log_session_ended(
        self,
        session_id: str,
        *,
        container_id: str | None = None,
        duration_seconds: float | None = None,
        command_count: int | None = None,
    ) -> None:
        """Log session end event.

        Args:
            session_id: Session identifier.
            container_id: Docker container ID if applicable.
            duration_seconds: Total session duration.
            command_count: Number of commands executed in session.
        """
        extra: dict[str, Any] = {
            "event_type": AuditEventType.SESSION_ENDED.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session_id,
        }
        if container_id:
            extra["container_id"] = container_id
        if duration_seconds is not None:
            extra["duration_seconds"] = round(duration_seconds, 2)
        if command_count is not None:
            extra["command_count"] = command_count
        if user := self._get_user():
            extra["user"] = user

        self._log(logging.INFO, "session_ended", **extra)


# =============================================================================
# Global Audit Logger
# =============================================================================


_audit_logger: ShellAuditLogger | None = None


def get_shell_audit_logger() -> ShellAuditLogger:
    """Get the global shell audit logger.

    Returns:
        The global ShellAuditLogger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = ShellAuditLogger()
    return _audit_logger


def set_shell_audit_logger(logger: ShellAuditLogger) -> None:
    """Set the global shell audit logger.

    Args:
        logger: The ShellAuditLogger to use globally.
    """
    global _audit_logger
    _audit_logger = logger
