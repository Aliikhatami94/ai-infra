"""Safety utilities for detecting destructive operations.

Phase 2.3.3: Pause before destructive operations.

This module provides pattern matching to detect potentially dangerous
operations in agent output, such as:
- File system deletions (rm -rf, shutil.rmtree)
- Database drops (DROP TABLE, TRUNCATE)
- Git force operations (push --force, reset --hard)
- Kubernetes bulk deletes (kubectl delete --all)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from typing import Any

logger = get_logger("executor.utils.safety")


# =============================================================================
# Destructive Operation Patterns
# =============================================================================

# File system patterns
FILE_SYSTEM_PATTERNS = [
    (r"rm\s+-rf\s+", "rm -rf (recursive force delete)"),
    (r"rm\s+-r\s+/", "rm -r / (recursive delete from root)"),
    (r"rm\s+--no-preserve-root", "rm --no-preserve-root"),
    (r"rmdir\s+/", "rmdir / (remove root directory)"),
    (r"shutil\.rmtree\s*\(", "shutil.rmtree (Python recursive delete)"),
    (r"os\.remove\s*\(\s*['\"]?/", "os.remove / (delete from root)"),
    (r"pathlib.*unlink.*missing_ok", "pathlib unlink"),
]

# Database patterns
DATABASE_PATTERNS = [
    (r"DROP\s+TABLE", "DROP TABLE"),
    (r"DROP\s+DATABASE", "DROP DATABASE"),
    (r"DROP\s+SCHEMA", "DROP SCHEMA"),
    (r"TRUNCATE\s+TABLE", "TRUNCATE TABLE"),
    (r"DELETE\s+FROM\s+\w+\s*;", "DELETE without WHERE clause"),
    (r"DELETE\s+FROM\s+\w+\s*$", "DELETE without WHERE clause"),
    (r"DROP\s+INDEX", "DROP INDEX"),
    (r"ALTER\s+TABLE.*DROP\s+COLUMN", "DROP COLUMN"),
]

# Git patterns
GIT_PATTERNS = [
    (r"git\s+push\s+.*--force", "git push --force"),
    (r"git\s+push\s+-f\s+", "git push -f (force)"),
    (r"git\s+reset\s+--hard", "git reset --hard"),
    (r"git\s+clean\s+-fd", "git clean -fd (force delete untracked)"),
    (r"git\s+clean\s+-f", "git clean -f"),
    (r"git\s+checkout\s+--\s+\.", "git checkout -- . (discard all changes)"),
    (r"git\s+branch\s+-D", "git branch -D (force delete branch)"),
    (r"git\s+rebase\s+.*--force", "git rebase --force"),
]

# Kubernetes patterns
KUBERNETES_PATTERNS = [
    (r"kubectl\s+delete\s+.*--all", "kubectl delete --all"),
    (r"kubectl\s+delete\s+namespace", "kubectl delete namespace"),
    (r"kubectl\s+delete\s+ns\s+", "kubectl delete ns"),
    (r"helm\s+uninstall\s+", "helm uninstall"),
    (r"kubectl\s+.*--force\s+--grace-period=0", "kubectl force delete"),
]

# Docker patterns
DOCKER_PATTERNS = [
    (r"docker\s+system\s+prune\s+-a", "docker system prune -a"),
    (r"docker\s+volume\s+rm\s+", "docker volume rm"),
    (r"docker\s+container\s+rm\s+.*-f", "docker container rm -f"),
    (r"docker-compose\s+down\s+-v", "docker-compose down -v (remove volumes)"),
]

# Cloud provider patterns
CLOUD_PATTERNS = [
    (r"aws\s+s3\s+rm\s+.*--recursive", "aws s3 rm --recursive"),
    (r"aws\s+ec2\s+terminate-instances", "aws ec2 terminate-instances"),
    (r"gcloud\s+.*delete", "gcloud delete"),
    (r"az\s+.*delete", "az delete"),
]

# System patterns
SYSTEM_PATTERNS = [
    (r"mkfs\.", "mkfs (format filesystem)"),
    (r"dd\s+if=.*of=/dev/", "dd to device (overwrite disk)"),
    (r">\s*/dev/sd[a-z]", "redirect to disk device"),
    (r"chmod\s+-R\s+777", "chmod -R 777 (insecure permissions)"),
    (r"chown\s+-R\s+.*:\s*/", "chown -R from root"),
]

# All patterns combined
ALL_DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = (
    FILE_SYSTEM_PATTERNS
    + DATABASE_PATTERNS
    + GIT_PATTERNS
    + KUBERNETES_PATTERNS
    + DOCKER_PATTERNS
    + CLOUD_PATTERNS
    + SYSTEM_PATTERNS
)


# =============================================================================
# Detection Result
# =============================================================================


@dataclass
class DestructiveOperation:
    """A detected destructive operation.

    Attributes:
        pattern: The regex pattern that matched.
        description: Human-readable description of the operation.
        match: The actual matched text.
        category: Category of the operation (filesystem, database, etc.).
    """

    pattern: str
    description: str
    match: str
    category: str


# =============================================================================
# Detection Functions
# =============================================================================


def detect_destructive_operations(
    content: str,
    *,
    include_categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
) -> list[DestructiveOperation]:
    """Detect potentially destructive operations in content.

    Scans the provided content for patterns that indicate dangerous operations
    such as file deletions, database drops, git force operations, etc.

    Args:
        content: The content to scan (agent output, code, commands).
        include_categories: Only check these categories (None = all).
        exclude_categories: Skip these categories.

    Returns:
        List of DestructiveOperation objects describing found operations.

    Example:
        >>> ops = detect_destructive_operations("rm -rf /tmp/data")
        >>> len(ops) > 0
        True
        >>> ops[0].category
        'filesystem'
    """
    if not content:
        return []

    found: list[DestructiveOperation] = []
    content.lower()

    # Category mapping
    category_patterns = {
        "filesystem": FILE_SYSTEM_PATTERNS,
        "database": DATABASE_PATTERNS,
        "git": GIT_PATTERNS,
        "kubernetes": KUBERNETES_PATTERNS,
        "docker": DOCKER_PATTERNS,
        "cloud": CLOUD_PATTERNS,
        "system": SYSTEM_PATTERNS,
    }

    for category, patterns in category_patterns.items():
        # Filter by include/exclude
        if include_categories and category not in include_categories:
            continue
        if exclude_categories and category in exclude_categories:
            continue

        for pattern, description in patterns:
            try:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    found.append(
                        DestructiveOperation(
                            pattern=pattern,
                            description=description,
                            match=match.group(),
                            category=category,
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
                continue

    # Deduplicate by match text
    seen_matches: set[str] = set()
    unique_found: list[DestructiveOperation] = []
    for op in found:
        if op.match not in seen_matches:
            seen_matches.add(op.match)
            unique_found.append(op)

    if unique_found:
        logger.warning(
            f"Detected {len(unique_found)} destructive operation(s): "
            f"{[op.description for op in unique_found]}"
        )

    return unique_found


def has_destructive_operations(content: str) -> bool:
    """Quick check if content contains any destructive operations.

    Args:
        content: The content to scan.

    Returns:
        True if any destructive operations were detected.
    """
    return len(detect_destructive_operations(content)) > 0


def format_destructive_warning(operations: list[DestructiveOperation]) -> str:
    """Format a human-readable warning message for detected operations.

    Args:
        operations: List of detected destructive operations.

    Returns:
        Formatted warning message suitable for display to user.
    """
    if not operations:
        return ""

    lines = ["Destructive operations detected:", ""]

    # Group by category
    by_category: dict[str, list[DestructiveOperation]] = {}
    for op in operations:
        if op.category not in by_category:
            by_category[op.category] = []
        by_category[op.category].append(op)

    for category, ops in sorted(by_category.items()):
        lines.append(f"  {category.upper()}:")
        for op in ops:
            lines.append(f"    - {op.description}")
            lines.append(f"      Match: {op.match[:80]}{'...' if len(op.match) > 80 else ''}")
        lines.append("")

    lines.append("These operations may cause data loss or system damage.")
    lines.append("Review carefully before proceeding.")

    return "\n".join(lines)


def check_agent_result_for_destructive_ops(
    result: Any,
) -> list[DestructiveOperation]:
    """Check an agent execution result for destructive operations.

    Handles various result types: dict, string, object with attributes.

    Args:
        result: The agent execution result.

    Returns:
        List of detected destructive operations.
    """
    content_to_check: list[str] = []

    # Handle dict results
    if isinstance(result, dict):
        # Check common fields
        for key in ("output", "content", "message", "response", "text", "code"):
            if key in result:
                content_to_check.append(str(result[key]))

        # Check tool calls
        tool_calls = result.get("tool_calls", [])
        for call in tool_calls:
            if isinstance(call, dict):
                content_to_check.append(str(call.get("args", {})))
                content_to_check.append(str(call.get("input", "")))

        # Check files modified
        for file_content in result.get("files_modified_content", []):
            content_to_check.append(str(file_content))

    # Handle string results
    elif isinstance(result, str):
        content_to_check.append(result)

    # Handle objects with common attributes
    else:
        for attr in ("output", "content", "message", "text"):
            if hasattr(result, attr):
                content_to_check.append(str(getattr(result, attr)))

    # Combine all content and check
    combined = "\n".join(content_to_check)
    return detect_destructive_operations(combined)


def check_files_for_destructive_ops(
    file_paths: list[str],
    workspace_root: str | None = None,
) -> list[DestructiveOperation]:
    """Check file contents on disk for destructive operations.

    This function reads the actual file content from disk and scans for
    destructive patterns. This catches cases where the agent's response
    text doesn't include the destructive pattern but the file does.

    Args:
        file_paths: List of file paths to check (can be relative or absolute).
        workspace_root: Optional workspace root for resolving relative paths.

    Returns:
        List of detected destructive operations with file path in match field.
    """
    from pathlib import Path

    all_ops: list[DestructiveOperation] = []

    for file_path in file_paths:
        try:
            # Resolve path
            path = Path(file_path)
            if not path.is_absolute() and workspace_root:
                path = Path(workspace_root) / file_path

            if not path.exists() or not path.is_file():
                continue

            # Read file content
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Skip binary files
                continue

            # Check for destructive operations
            ops = detect_destructive_operations(content)

            # Add file path context to each operation
            for op in ops:
                all_ops.append(
                    DestructiveOperation(
                        pattern=op.pattern,
                        description=op.description,
                        match=f"{file_path}: {op.match}",
                        category=op.category,
                    )
                )

        except Exception as e:
            logger.debug(f"Could not check file {file_path}: {e}")
            continue

    if all_ops:
        logger.warning(
            f"Detected {len(all_ops)} destructive operation(s) in files: "
            f"{[op.description for op in all_ops]}"
        )

    return all_ops
