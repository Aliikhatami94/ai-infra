"""Utility modules for the executor.

Phase 2.3.3: Safety utilities for destructive operation detection.
"""

from ai_infra.executor.utils.safety import (
    DestructiveOperation,
    check_agent_result_for_destructive_ops,
    detect_destructive_operations,
    format_destructive_warning,
    has_destructive_operations,
)

__all__ = [
    "DestructiveOperation",
    "check_agent_result_for_destructive_ops",
    "detect_destructive_operations",
    "format_destructive_warning",
    "has_destructive_operations",
]
