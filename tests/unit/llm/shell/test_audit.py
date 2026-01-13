"""Tests for shell audit logging module.

Phase 4.4 of EXECUTOR_CLI.md - Audit Logging.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from ai_infra.llm.shell.audit import (
    AuditEvent,
    AuditEventType,
    RedactionEvent,
    SecurityViolationEvent,
    ShellAuditLogger,
    get_shell_audit_logger,
    set_shell_audit_logger,
)
from ai_infra.llm.shell.security import ValidationResult, ValidationStatus
from ai_infra.llm.shell.types import ShellResult

# =============================================================================
# Test AuditEventType
# =============================================================================


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_event_types_exist(self) -> None:
        """All expected event types should exist."""
        assert AuditEventType.COMMAND_EXECUTED.value == "command_executed"
        assert AuditEventType.COMMAND_FAILED.value == "command_failed"
        assert AuditEventType.COMMAND_TIMEOUT.value == "command_timeout"
        assert AuditEventType.SECRET_REDACTED.value == "secret_redacted"
        assert AuditEventType.SECURITY_VIOLATION.value == "security_violation"
        assert AuditEventType.SESSION_STARTED.value == "session_started"
        assert AuditEventType.SESSION_ENDED.value == "session_ended"


# =============================================================================
# Test AuditEvent
# =============================================================================


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_basic_event(self) -> None:
        """Basic event should store all fields."""
        event = AuditEvent(
            event_type=AuditEventType.COMMAND_EXECUTED,
            timestamp="2024-01-15T10:30:00Z",
            command="echo hello",
            exit_code=0,
            duration_ms=5.2,
            success=True,
        )
        assert event.event_type == AuditEventType.COMMAND_EXECUTED
        assert event.command == "echo hello"
        assert event.exit_code == 0
        assert event.success is True

    def test_to_dict(self) -> None:
        """to_dict should include all non-None fields."""
        event = AuditEvent(
            event_type=AuditEventType.COMMAND_EXECUTED,
            timestamp="2024-01-15T10:30:00Z",
            command="echo hello",
            exit_code=0,
            duration_ms=5.234,
            success=True,
            session_id="sess-123",
        )
        d = event.to_dict()

        assert d["event_type"] == "command_executed"
        assert d["command"] == "echo hello"
        assert d["exit_code"] == 0
        assert d["duration_ms"] == 5.23  # Rounded
        assert d["session_id"] == "sess-123"

    def test_to_dict_excludes_none(self) -> None:
        """to_dict should exclude None optional fields."""
        event = AuditEvent(
            event_type=AuditEventType.COMMAND_EXECUTED,
            timestamp="2024-01-15T10:30:00Z",
            command="echo hello",
        )
        d = event.to_dict()

        assert "session_id" not in d
        assert "container_id" not in d
        assert "user" not in d

    def test_from_result_success(self) -> None:
        """from_result should create event from successful ShellResult."""
        result = ShellResult(
            success=True,
            exit_code=0,
            stdout="hello",
            stderr="",
            command="echo hello",
            duration_ms=5.2,
        )
        event = AuditEvent.from_result(result)

        assert event.event_type == AuditEventType.COMMAND_EXECUTED
        assert event.success is True
        assert event.exit_code == 0

    def test_from_result_failure(self) -> None:
        """from_result should create event from failed ShellResult."""
        result = ShellResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="error",
            command="false",
            duration_ms=5.2,
        )
        event = AuditEvent.from_result(result)

        assert event.event_type == AuditEventType.COMMAND_FAILED
        assert event.success is False
        assert event.exit_code == 1

    def test_from_result_timeout(self) -> None:
        """from_result should create timeout event from timed-out ShellResult."""
        result = ShellResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="timeout",
            command="sleep 100",
            duration_ms=30000,
            timed_out=True,
        )
        event = AuditEvent.from_result(result)

        assert event.event_type == AuditEventType.COMMAND_TIMEOUT
        assert event.success is False

    def test_from_result_with_session(self) -> None:
        """from_result should include session and container IDs."""
        result = ShellResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            command="echo",
            duration_ms=1.0,
        )
        event = AuditEvent.from_result(
            result,
            session_id="sess-123",
            container_id="abc123",
            user="testuser",
        )

        assert event.session_id == "sess-123"
        assert event.container_id == "abc123"
        assert event.user == "testuser"


# =============================================================================
# Test RedactionEvent
# =============================================================================


class TestRedactionEvent:
    """Tests for RedactionEvent dataclass."""

    def test_basic_event(self) -> None:
        """Basic redaction event should store fields."""
        event = RedactionEvent(
            timestamp="2024-01-15T10:30:00Z",
            rule_name="openai_api_key",
            count=2,
            command="echo $API_KEY",
        )
        assert event.rule_name == "openai_api_key"
        assert event.count == 2

    def test_to_dict(self) -> None:
        """to_dict should include event_type."""
        event = RedactionEvent(
            timestamp="2024-01-15T10:30:00Z",
            rule_name="openai_api_key",
            count=1,
            command="echo",
        )
        d = event.to_dict()

        assert d["event_type"] == "secret_redacted"
        assert d["rule_name"] == "openai_api_key"
        assert d["count"] == 1


# =============================================================================
# Test SecurityViolationEvent
# =============================================================================


class TestSecurityViolationEvent:
    """Tests for SecurityViolationEvent dataclass."""

    def test_basic_event(self) -> None:
        """Basic violation event should store fields."""
        event = SecurityViolationEvent(
            timestamp="2024-01-15T10:30:00Z",
            command="rm -rf /",
            reason="Matches denied pattern",
            policy_name="strict",
            matched_pattern=r"rm\s+-rf",
        )
        assert event.command == "rm -rf /"
        assert event.reason == "Matches denied pattern"
        assert event.policy_name == "strict"

    def test_to_dict(self) -> None:
        """to_dict should include all fields."""
        event = SecurityViolationEvent(
            timestamp="2024-01-15T10:30:00Z",
            command="rm -rf /",
            reason="Denied",
            policy_name="strict",
            matched_pattern=r"rm\s+-rf",
            user="attacker",
        )
        d = event.to_dict()

        assert d["event_type"] == "security_violation"
        assert d["command"] == "rm -rf /"
        assert d["reason"] == "Denied"
        assert d["matched_pattern"] == r"rm\s+-rf"
        assert d["user"] == "attacker"

    def test_from_validation_result(self) -> None:
        """from_validation_result should create event from ValidationResult."""
        result = ValidationResult(
            status=ValidationStatus.DENIED,
            command="rm -rf /",
            reason="Command matches denied pattern",
            matched_pattern=r"rm\s+-rf",
        )
        event = SecurityViolationEvent.from_validation_result(
            command="rm -rf /",
            result=result,
            policy_name="production",
        )

        assert event.reason == "Command matches denied pattern"
        assert event.matched_pattern == r"rm\s+-rf"
        assert event.policy_name == "production"


# =============================================================================
# Test ShellAuditLogger
# =============================================================================


class TestShellAuditLogger:
    """Tests for ShellAuditLogger."""

    def test_default_initialization(self) -> None:
        """Default logger should be enabled."""
        logger = ShellAuditLogger()
        assert logger.enabled is True

    def test_disabled_logger(self) -> None:
        """Disabled logger should not log."""
        logger = ShellAuditLogger(enabled=False)
        assert logger.enabled is False

    def test_enable_disable(self) -> None:
        """Should be able to toggle enabled state."""
        logger = ShellAuditLogger()
        logger.enabled = False
        assert logger.enabled is False
        logger.enabled = True
        assert logger.enabled is True

    def test_log_command_success(self) -> None:
        """log_command should log successful commands at INFO level."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_command(
                    command="echo hello",
                    exit_code=0,
                    duration_ms=5.2,
                )

        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        assert args[0] == logging.INFO
        assert args[1] == "shell_command"
        assert kwargs["extra"]["event_type"] == "command_executed"
        assert kwargs["extra"]["success"] is True

    def test_log_command_failure(self) -> None:
        """log_command should log failed commands at WARNING level."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_command(
                    command="false",
                    exit_code=1,
                    duration_ms=5.2,
                )

        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        assert args[0] == logging.WARNING
        assert kwargs["extra"]["event_type"] == "command_failed"

    def test_log_command_timeout(self) -> None:
        """log_command should log timeouts at WARNING level."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_command(
                    command="sleep 100",
                    exit_code=-1,
                    duration_ms=30000,
                    timed_out=True,
                )

        args, kwargs = mock_log.call_args
        assert args[0] == logging.WARNING
        assert kwargs["extra"]["event_type"] == "command_timeout"

    def test_log_result(self) -> None:
        """log_result should extract fields from ShellResult."""
        logger = ShellAuditLogger()
        result = ShellResult(
            success=True,
            exit_code=0,
            stdout="hello",
            stderr="",
            command="echo hello",
            duration_ms=5.2,
        )

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_result(result, session_id="sess-123")

        args, kwargs = mock_log.call_args
        assert kwargs["extra"]["command"] == "echo hello"
        assert kwargs["extra"]["session_id"] == "sess-123"

    def test_log_redaction(self) -> None:
        """log_redaction should log at WARNING level."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_redaction(rule_name="openai_api_key", count=2)

        args, kwargs = mock_log.call_args
        assert args[0] == logging.WARNING
        assert args[1] == "secret_redacted"
        assert kwargs["extra"]["rule_name"] == "openai_api_key"
        assert kwargs["extra"]["count"] == 2

    def test_log_redactions_multiple(self) -> None:
        """log_redactions should log multiple rules."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_redactions(
                    {
                        "openai_api_key": 1,
                        "aws_secret_key": 2,
                        "empty_rule": 0,  # Should not be logged
                    }
                )

        # Should be called twice (skip zero counts)
        assert mock_log.call_count == 2

    def test_log_security_violation(self) -> None:
        """log_security_violation should log at WARNING level."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_security_violation(
                    command="rm -rf /",
                    reason="Dangerous command",
                    matched_pattern=r"rm\s+-rf",
                )

        args, kwargs = mock_log.call_args
        assert args[0] == logging.WARNING
        assert args[1] == "security_violation"
        assert kwargs["extra"]["reason"] == "Dangerous command"

    def test_log_validation_result_allowed(self) -> None:
        """log_validation_result should not log allowed commands."""
        logger = ShellAuditLogger()
        result = ValidationResult(
            status=ValidationStatus.ALLOWED,
            command="echo hello",
            reason="Command is allowed",
        )

        with patch.object(logger._logger, "log") as mock_log:
            logger.log_validation_result("echo hello", result)

        mock_log.assert_not_called()

    def test_log_validation_result_denied(self) -> None:
        """log_validation_result should log denied commands."""
        logger = ShellAuditLogger()
        result = ValidationResult(
            status=ValidationStatus.DENIED,
            command="rm -rf /",
            reason="Denied",
        )

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_validation_result("rm -rf /", result)

        mock_log.assert_called_once()

    def test_log_session_started(self) -> None:
        """log_session_started should log at INFO level."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_session_started(
                    session_id="sess-123",
                    container_id="abc123",
                )

        args, kwargs = mock_log.call_args
        assert args[0] == logging.INFO
        assert args[1] == "session_started"
        assert kwargs["extra"]["session_id"] == "sess-123"
        assert kwargs["extra"]["container_id"] == "abc123"

    def test_log_session_ended(self) -> None:
        """log_session_ended should log at INFO level."""
        logger = ShellAuditLogger()

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_session_ended(
                    session_id="sess-123",
                    duration_seconds=120.5,
                    command_count=10,
                )

        args, kwargs = mock_log.call_args
        assert args[0] == logging.INFO
        assert args[1] == "session_ended"
        assert kwargs["extra"]["duration_seconds"] == 120.5
        assert kwargs["extra"]["command_count"] == 10

    def test_disabled_logger_no_log(self) -> None:
        """Disabled logger should not make any log calls."""
        logger = ShellAuditLogger(enabled=False)

        with patch.object(logger._logger, "log") as mock_log:
            logger.log_command("echo", 0, 1.0)
            logger.log_redaction("rule", 1)
            logger.log_security_violation("cmd", "reason")

        mock_log.assert_not_called()

    def test_command_truncation(self) -> None:
        """Long commands should be truncated."""
        logger = ShellAuditLogger()
        long_command = "A" * 500

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_command(long_command, 0, 1.0)

        args, kwargs = mock_log.call_args
        logged_command = kwargs["extra"]["command"]
        assert len(logged_command) <= 203  # 200 + "..."
        assert logged_command.endswith("...")

    def test_include_user_disabled(self) -> None:
        """Logger with include_user=False should not include user."""
        logger = ShellAuditLogger(include_user=False)

        with patch.object(logger._logger, "log") as mock_log:
            with patch.object(logger._logger, "isEnabledFor", return_value=True):
                logger.log_command("echo", 0, 1.0)

        args, kwargs = mock_log.call_args
        assert "user" not in kwargs["extra"] or kwargs["extra"].get("user") is None


# =============================================================================
# Test Global Logger Functions
# =============================================================================


class TestGlobalAuditLogger:
    """Tests for global audit logger functions."""

    def test_get_shell_audit_logger(self) -> None:
        """get_shell_audit_logger should return a logger instance."""
        logger = get_shell_audit_logger()
        assert isinstance(logger, ShellAuditLogger)

    def test_get_shell_audit_logger_singleton(self) -> None:
        """get_shell_audit_logger should return same instance."""
        logger1 = get_shell_audit_logger()
        logger2 = get_shell_audit_logger()
        assert logger1 is logger2

    def test_set_shell_audit_logger(self) -> None:
        """set_shell_audit_logger should replace global logger."""
        custom_logger = ShellAuditLogger(name="custom")
        set_shell_audit_logger(custom_logger)

        retrieved = get_shell_audit_logger()
        assert retrieved is custom_logger

        # Reset to default
        set_shell_audit_logger(ShellAuditLogger())


# =============================================================================
# Integration Tests
# =============================================================================


class TestAuditIntegration:
    """Integration tests for audit logging."""

    def test_full_command_flow(self) -> None:
        """Test complete flow of logging a command."""
        logger = ShellAuditLogger()
        result = ShellResult(
            success=True,
            exit_code=0,
            stdout="hello world",
            stderr="",
            command="echo hello world",
            duration_ms=2.5,
        )

        # This should not raise
        logger.log_result(result, session_id="test-session")

    def test_full_security_flow(self) -> None:
        """Test complete flow of logging a security violation."""
        logger = ShellAuditLogger()
        validation = ValidationResult(
            status=ValidationStatus.DENIED,
            command="rm -rf /",
            reason="Command matches denied pattern: rm -rf",
            matched_pattern=r"rm\s+-rf\s+/",
        )

        # This should not raise
        logger.log_validation_result(
            "rm -rf /",
            validation,
            policy_name="production",
        )

    def test_session_lifecycle(self) -> None:
        """Test logging session start and end."""
        logger = ShellAuditLogger()

        # This should not raise
        logger.log_session_started(
            session_id="sess-abc",
            container_id="container-123",
            config={"image": "python:3.11-slim"},
        )

        logger.log_session_ended(
            session_id="sess-abc",
            container_id="container-123",
            duration_seconds=300.0,
            command_count=25,
        )
