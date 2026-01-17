"""Security module for shell command validation.

This module provides security controls for shell command execution:

- **SecurityPolicy**: Configuration for command filtering
- **validate_command**: Check if a command is allowed
- **DEFAULT_DENIED_PATTERNS**: Default dangerous command patterns
- **DEFAULT_ALLOWED_PATTERNS**: Default safe command patterns

Phase 4.1 of EXECUTOR_CLI.md - Security & Safety.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "SecurityPolicy",
    "ValidationResult",
    "ValidationStatus",
    "validate_command",
    "validate_command_with_network",
    "is_network_command",
    "DEFAULT_DENIED_PATTERNS",
    "DEFAULT_ALLOWED_PATTERNS",
    "compile_patterns",
    "create_strict_policy",
    "create_permissive_policy",
]


# =============================================================================
# Default Security Patterns
# =============================================================================

# Patterns for commands that should NEVER be allowed
DEFAULT_DENIED_PATTERNS: tuple[str, ...] = (
    # Destructive filesystem commands
    r"rm\s+(-[rRfvi]+\s+)*/([\s]|$)",  # rm -rf /
    r"rm\s+(-[rRfvi]+\s+)*/\*",  # rm -rf /*
    r"rm\s+(-[rRfvi]+\s+)*~([\s]|$)",  # rm -rf ~
    r"rm\s+(-[rRfvi]+\s+)*\.\.([\s]|$)",  # rm -rf ..
    # Fork bomb
    r":\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    # Filesystem formatting
    r"mkfs(\.[a-z0-9]+)?(\s|$)",
    r"dd\s+.*of=/dev/",
    # Remote code execution
    r"curl\s+.*\|\s*(ba)?sh",
    r"wget\s+.*\|\s*(ba)?sh",
    r"fetch\s+.*\|\s*(ba)?sh",
    # Sensitive file modification
    r">\s*/etc/passwd",
    r">\s*/etc/shadow",
    r">\s*/etc/sudoers",
    r"tee\s+/etc/passwd",
    r"tee\s+/etc/shadow",
    # Dangerous permissions
    r"chmod\s+777\s+/",
    r"chmod\s+-R\s+777\s+/",
    r"chown\s+-R\s+.*\s+/",
    # Privilege escalation (when allow_sudo=False)
    r"sudo\s+",
    r"su\s+-",
    r"doas\s+",
    # History/credential access
    r"cat\s+.*\.bash_history",
    r"cat\s+.*\.ssh/",
    r"cat\s+.*\.aws/credentials",
    r"cat\s+.*\.netrc",
    # Reverse shells
    r"bash\s+-i\s+>&\s+/dev/tcp/",
    r"nc\s+.*-e\s+/bin/(ba)?sh",
    r"python.*socket.*connect",
    # System shutdown/reboot
    r"shutdown(\s|$)",
    r"reboot(\s|$)",
    r"init\s+[06]",
    r"systemctl\s+(halt|poweroff|reboot)",
)

# Patterns for commands that are typically safe for development workflows
DEFAULT_ALLOWED_PATTERNS: tuple[str, ...] = (
    # Testing
    r"pytest\s*.*",
    r"python\s+-m\s+pytest\s*.*",
    r"npm\s+(test|run\s+test)\s*.*",
    r"yarn\s+(test|run\s+test)\s*.*",
    r"pnpm\s+(test|run\s+test)\s*.*",
    r"cargo\s+test\s*.*",
    r"go\s+test\s*.*",
    r"jest\s*.*",
    r"vitest\s*.*",
    r"mocha\s*.*",
    # Build tools
    r"make\s+\w+",
    r"npm\s+(run|install|build|ci)\s*.*",
    r"yarn\s+(install|build|add)\s*.*",
    r"pnpm\s+(install|build|add)\s*.*",
    r"cargo\s+(build|check|clippy)\s*.*",
    r"go\s+(build|mod)\s*.*",
    r"tsc(\s+.*)?$",
    # Python package management
    r"poetry\s+(run|install|add|update|lock)\s*.*",
    r"pip\s+install\s*.*",
    r"pip\s+freeze\s*.*",
    r"pip\s+list\s*.*",
    r"uv\s+(pip|sync|lock)\s*.*",
    # Linting and formatting
    r"ruff\s+(check|format)\s*.*",
    r"black\s+.*",
    r"isort\s+.*",
    r"mypy\s+.*",
    r"flake8\s+.*",
    r"eslint\s+.*",
    r"prettier\s+.*",
    r"rustfmt\s+.*",
    # Safe file operations
    r"cat\s+[^/].*",  # cat for non-root files
    r"head\s+.*",
    r"tail\s+.*",
    r"grep\s+.*",
    r"find\s+\.\s+.*",  # find in current directory
    r"ls\s*.*",
    r"pwd\s*$",
    r"echo\s+.*",
    r"printf\s+.*",
    r"wc\s+.*",
    r"sort\s+.*",
    r"uniq\s+.*",
    r"diff\s+.*",
    # Git (read operations)
    r"git\s+(status|log|diff|show|branch|remote|fetch)\s*.*",
    r"git\s+ls-files\s*.*",
    # Directory operations
    r"mkdir\s+(-p\s+)?[^/].*",  # mkdir for non-root paths
    r"cd\s+.*",
    r"tree\s+.*",
    # Environment inspection
    r"env\s*$",
    r"printenv\s*.*",
    r"which\s+.*",
    r"type\s+.*",
    r"command\s+-v\s+.*",
)


# =============================================================================
# Validation Types
# =============================================================================


class ValidationStatus(Enum):
    """Status of command validation."""

    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validating a command.

    Attributes:
        status: Whether the command is allowed, denied, or needs approval.
        command: The command that was validated.
        reason: Human-readable explanation of the decision.
        matched_pattern: The pattern that matched (if any).
    """

    status: ValidationStatus
    command: str
    reason: str
    matched_pattern: str | None = None

    @property
    def is_allowed(self) -> bool:
        """Check if the command is allowed to execute."""
        return self.status == ValidationStatus.ALLOWED

    @property
    def is_denied(self) -> bool:
        """Check if the command is denied."""
        return self.status == ValidationStatus.DENIED


# =============================================================================
# Security Policy
# =============================================================================


@dataclass
class SecurityPolicy:
    """Configuration for command security validation.

    Attributes:
        allowed_patterns: Regex patterns for allowed commands.
        denied_patterns: Regex patterns for denied commands.
        allow_sudo: Whether to allow sudo commands (default: False).
        allow_network: Whether to allow network access (default: True).
        max_file_writes: Maximum number of file write operations (default: 100).
        strict_mode: If True, only explicitly allowed commands are permitted.
            If False (default), commands not matching denied patterns are allowed.
        custom_denied_commands: Specific commands to deny (exact match).
        custom_allowed_commands: Specific commands to allow (exact match).

    Example:
        >>> policy = SecurityPolicy(
        ...     allow_sudo=False,
        ...     strict_mode=True,
        ... )
        >>> result = validate_command("pytest -v", policy)
        >>> result.is_allowed
        True
    """

    allowed_patterns: tuple[str, ...] = DEFAULT_ALLOWED_PATTERNS
    denied_patterns: tuple[str, ...] = DEFAULT_DENIED_PATTERNS
    allow_sudo: bool = False
    allow_network: bool = True
    max_file_writes: int = 100
    strict_mode: bool = False
    custom_denied_commands: tuple[str, ...] = ()
    custom_allowed_commands: tuple[str, ...] = ()

    # Compiled patterns (populated lazily)
    _compiled_allowed: tuple[re.Pattern[str], ...] | None = field(
        default=None, repr=False, compare=False
    )
    _compiled_denied: tuple[re.Pattern[str], ...] | None = field(
        default=None, repr=False, compare=False
    )

    def get_compiled_allowed(self) -> tuple[re.Pattern[str], ...]:
        """Get compiled allowed patterns (cached)."""
        if self._compiled_allowed is None:
            object.__setattr__(self, "_compiled_allowed", compile_patterns(self.allowed_patterns))
        return self._compiled_allowed  # type: ignore[return-value]

    def get_compiled_denied(self) -> tuple[re.Pattern[str], ...]:
        """Get compiled denied patterns (cached)."""
        if self._compiled_denied is None:
            # Filter sudo pattern if allow_sudo is True
            patterns = self.denied_patterns
            if self.allow_sudo:
                patterns = tuple(
                    p
                    for p in patterns
                    if not p.startswith(r"sudo\s+")
                    and not p.startswith(r"su\s+-")
                    and not p.startswith(r"doas\s+")
                )
            object.__setattr__(self, "_compiled_denied", compile_patterns(patterns))
        return self._compiled_denied  # type: ignore[return-value]


# =============================================================================
# Pattern Compilation
# =============================================================================


def compile_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile regex patterns for matching.

    Args:
        patterns: Tuple of regex pattern strings.

    Returns:
        Tuple of compiled regex patterns.
    """
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# =============================================================================
# Command Validation
# =============================================================================


def validate_command(command: str, policy: SecurityPolicy | None = None) -> ValidationResult:
    """Validate a shell command against a security policy.

    This function checks if a command should be allowed to execute based on
    the provided security policy. It performs the following checks in order:

    1. Check if command exactly matches custom denied commands
    2. Check if command exactly matches custom allowed commands
    3. Check if command matches any denied patterns
    4. If strict_mode is enabled, check if command matches allowed patterns
    5. If not strict_mode, allow commands that don't match denied patterns

    Args:
        command: The shell command to validate.
        policy: Security policy to apply. If None, uses default policy.

    Returns:
        ValidationResult indicating whether the command is allowed, denied,
        or requires approval.

    Example:
        >>> policy = SecurityPolicy()
        >>> result = validate_command("pytest -v tests/", policy)
        >>> result.is_allowed
        True
        >>> result = validate_command("rm -rf /", policy)
        >>> result.is_denied
        True
    """
    if policy is None:
        policy = SecurityPolicy()

    # Normalize command (strip whitespace)
    command = command.strip()

    if not command:
        return ValidationResult(
            status=ValidationStatus.DENIED,
            command=command,
            reason="Empty command",
        )

    # Check custom denied commands (exact match)
    for denied_cmd in policy.custom_denied_commands:
        if command == denied_cmd or command.startswith(denied_cmd + " "):
            return ValidationResult(
                status=ValidationStatus.DENIED,
                command=command,
                reason=f"Command matches custom denied command: {denied_cmd}",
                matched_pattern=denied_cmd,
            )

    # Check custom allowed commands (exact match) - takes precedence over patterns
    for allowed_cmd in policy.custom_allowed_commands:
        if command == allowed_cmd or command.startswith(allowed_cmd + " "):
            return ValidationResult(
                status=ValidationStatus.ALLOWED,
                command=command,
                reason=f"Command matches custom allowed command: {allowed_cmd}",
                matched_pattern=allowed_cmd,
            )

    # Check denied patterns
    for pattern in policy.get_compiled_denied():
        if pattern.search(command):
            return ValidationResult(
                status=ValidationStatus.DENIED,
                command=command,
                reason=f"Command matches denied pattern: {pattern.pattern}",
                matched_pattern=pattern.pattern,
            )

    # In strict mode, command must match an allowed pattern
    if policy.strict_mode:
        for pattern in policy.get_compiled_allowed():
            if pattern.search(command):
                return ValidationResult(
                    status=ValidationStatus.ALLOWED,
                    command=command,
                    reason=f"Command matches allowed pattern: {pattern.pattern}",
                    matched_pattern=pattern.pattern,
                )

        # No allowed pattern matched in strict mode
        return ValidationResult(
            status=ValidationStatus.REQUIRES_APPROVAL,
            command=command,
            reason="Command does not match any allowed pattern (strict mode enabled)",
        )

    # Non-strict mode: allow commands that don't match denied patterns
    # Check if it matches an allowed pattern for informational purposes
    for pattern in policy.get_compiled_allowed():
        if pattern.search(command):
            return ValidationResult(
                status=ValidationStatus.ALLOWED,
                command=command,
                reason=f"Command matches allowed pattern: {pattern.pattern}",
                matched_pattern=pattern.pattern,
            )

    # Command doesn't match any denied pattern, allow it
    return ValidationResult(
        status=ValidationStatus.ALLOWED,
        command=command,
        reason="Command does not match any denied pattern",
    )


def validate_command_with_network(
    command: str,
    policy: SecurityPolicy | None = None,
) -> ValidationResult:
    """Validate a command including network access checks.

    This extends validate_command to also check if the command attempts
    network access when policy.allow_network is False.

    Args:
        command: The shell command to validate.
        policy: Security policy to apply. If None, uses default policy.

    Returns:
        ValidationResult indicating whether the command is allowed.

    Example:
        >>> policy = SecurityPolicy(allow_network=False)
        >>> result = validate_command_with_network("curl https://api.example.com", policy)
        >>> result.is_denied
        True
        >>> result.reason
        'Network access is not allowed by security policy'
    """
    if policy is None:
        policy = SecurityPolicy()

    # First, run standard validation
    result = validate_command(command, policy)

    # If already denied, return early
    if result.is_denied:
        return result

    # Check network access if not allowed
    if not policy.allow_network and is_network_command(command):
        return ValidationResult(
            status=ValidationStatus.DENIED,
            command=command,
            reason="Network access is not allowed by security policy",
            matched_pattern="network_command",
        )

    return result


def is_network_command(command: str) -> bool:
    """Check if a command involves network access.

    Args:
        command: Shell command to check.

    Returns:
        True if the command appears to access the network.
    """
    network_patterns = (
        r"\bcurl\s+",
        r"\bwget\s+",
        r"\bfetch\s+",
        r"\bnc\s+",
        r"\bnetcat\s+",
        r"\bssh\s+",
        r"\bscp\s+",
        r"\brsync\s+",
        r"\bftp\s+",
        r"\bsftp\s+",
        r"\btelnet\s+",
        r"\bnmap\s+",
        r"\bping\s+",
        r"\btraceroute\s+",
        r"\bdig\s+",
        r"\bnslookup\s+",
        r"\bhost\s+",
        r"\bhttp\s+",  # httpie
        r"\bhttps?\://",  # URLs in commands
    )
    for pattern in network_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False


def create_strict_policy(
    allowed_commands: tuple[str, ...] = (),
    allowed_patterns: tuple[str, ...] | None = None,
) -> SecurityPolicy:
    """Create a strict security policy for production use.

    In strict mode, only explicitly allowed commands and patterns are permitted.
    This is the recommended mode for production environments.

    Args:
        allowed_commands: Specific commands to allow (exact match).
        allowed_patterns: Regex patterns for allowed commands.
            Defaults to DEFAULT_ALLOWED_PATTERNS.

    Returns:
        SecurityPolicy configured for strict mode.

    Example:
        >>> policy = create_strict_policy(
        ...     allowed_commands=("pytest", "make test"),
        ... )
        >>> validate_command("pytest -v", policy).is_allowed
        True
        >>> validate_command("curl https://example.com", policy).is_denied
        False  # REQUIRES_APPROVAL, not denied
    """
    return SecurityPolicy(
        allowed_patterns=allowed_patterns or DEFAULT_ALLOWED_PATTERNS,
        denied_patterns=DEFAULT_DENIED_PATTERNS,
        allow_sudo=False,
        allow_network=False,
        strict_mode=True,
        custom_allowed_commands=allowed_commands,
    )


def create_permissive_policy(
    denied_commands: tuple[str, ...] = (),
    denied_patterns: tuple[str, ...] | None = None,
) -> SecurityPolicy:
    """Create a permissive security policy for development use.

    In permissive mode, commands are allowed unless they match denied patterns.
    This is suitable for local development but NOT recommended for production.

    Args:
        denied_commands: Specific commands to deny (exact match).
        denied_patterns: Regex patterns for denied commands.
            Defaults to DEFAULT_DENIED_PATTERNS.

    Returns:
        SecurityPolicy configured for permissive mode.

    Example:
        >>> policy = create_permissive_policy()
        >>> validate_command("any-command", policy).is_allowed
        True  # Unless it matches a denied pattern
    """
    return SecurityPolicy(
        allowed_patterns=DEFAULT_ALLOWED_PATTERNS,
        denied_patterns=denied_patterns or DEFAULT_DENIED_PATTERNS,
        allow_sudo=False,
        allow_network=True,
        strict_mode=False,
        custom_denied_commands=denied_commands,
    )
