"""Deprecation utilities for ai-infra.

This module provides decorators and functions for marking features as deprecated,
following the deprecation policy defined in DEPRECATION.md.
"""

from ai_infra.utils import DeprecatedWarning, deprecated, deprecated_parameter

__all__ = [
    "deprecated",
    "deprecated_parameter",
    "DeprecatedWarning",
]
