"""PII leakage detection guardrail for LLM outputs.

This module provides guardrails for detecting when LLMs output sensitive
PII data that should not be exposed to users, such as SSNs, credit card
numbers, API keys, and other credentials.

Example:
    >>> from ai_infra.guardrails.output import PIILeakage
    >>>
    >>> guard = PIILeakage(entities=["SSN", "CREDIT_CARD", "API_KEY"])
    >>> result = guard.check(llm_output)
    >>> if not result.passed:
    ...     print(f"PII leaked: {result.reason}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from ai_infra.guardrails.base import Guardrail, GuardrailResult

logger = logging.getLogger(__name__)


@dataclass
class PIILeakageMatch:
    """Represents a detected PII leak in output."""

    entity_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0

    def __repr__(self) -> str:
        """Return a masked string representation."""
        masked = self.value[:4] + "..." if len(self.value) > 6 else "***"
        return f"PIILeakageMatch({self.entity_type}, '{masked}')"


class PIILeakage(Guardrail):
    """Guardrail for detecting PII leakage in LLM outputs.

    Detects when LLMs output sensitive information that should not
    be exposed, such as Social Security Numbers, credit card numbers,
    API keys, passwords, and other credentials.

    Unlike input PII detection (which validates user input), this guardrail
    focuses on preventing the LLM from inadvertently outputting sensitive
    data from its training or context.

    Attributes:
        name: "pii_leakage"
        entities: List of PII types to detect
        action: What to do when PII is found ("redact", "block", "warn")

    Example:
        >>> guard = PIILeakage(
        ...     entities=["SSN", "CREDIT_CARD", "API_KEY"],
        ...     action="redact"
        ... )
        >>> result = guard.check(llm_output)
        >>> if "redacted_text" in result.details:
        ...     safe_output = result.details["redacted_text"]
    """

    name = "pii_leakage"

    # PII patterns optimized for detecting leakage in LLM outputs
    PII_PATTERNS: dict[str, tuple[str, float]] = {
        "SSN": (
            r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b",
            0.95,
        ),
        "CREDIT_CARD": (
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            0.95,
        ),
        "API_KEY": (
            r"\b(?:sk-[a-zA-Z0-9]{32,}|AIza[0-9A-Za-z_-]{35}|ghp_[a-zA-Z0-9]{36}|glpat-[a-zA-Z0-9_-]{20,})\b",
            0.99,
        ),
        "AWS_ACCESS_KEY": (
            r"\bAKIA[0-9A-Z]{16}\b",
            0.99,
        ),
        "AWS_SECRET_KEY": (
            r"\b[A-Za-z0-9/+=]{40}\b",
            0.6,  # Lower confidence due to potential false positives
        ),
        "PRIVATE_KEY": (
            r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
            0.99,
        ),
        "PASSWORD": (
            r"\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^'\"\s]{8,})['\"]?",
            0.8,
        ),
        "BEARER_TOKEN": (
            r"\bBearer\s+[a-zA-Z0-9_-]{20,}\b",
            0.95,
        ),
        "JWT": (
            r"\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b",
            0.99,
        ),
        "EMAIL": (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            0.9,
        ),
        "PHONE": (
            r"(?:\+?1[-.\s]?)?(?:\([2-9]\d{2}\)[-.\s]?|[2-9]\d{2}[-.\s]?)\d{3}[-.\s]?\d{4}",
            0.85,
        ),
        "IP_ADDRESS": (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            0.8,
        ),
        "BANK_ACCOUNT": (
            r"\b\d{8,17}\b",  # Generic bank account number
            0.3,  # Low confidence, needs context
        ),
        "ROUTING_NUMBER": (
            r"\b(?:0[1-9]|[1-2][0-9]|3[0-2])\d{7}\b",
            0.7,
        ),
    }

    # Default entities for output scanning (focused on credentials and financial)
    DEFAULT_ENTITIES = [
        "SSN",
        "CREDIT_CARD",
        "API_KEY",
        "AWS_ACCESS_KEY",
        "PRIVATE_KEY",
        "PASSWORD",
        "BEARER_TOKEN",
        "JWT",
    ]

    def __init__(
        self,
        entities: list[str] | None = None,
        action: Literal["redact", "block", "warn"] = "redact",
        custom_patterns: dict[str, str] | None = None,
        min_confidence: float = 0.5,
    ):
        """Initialize the PII leakage detection guardrail.

        Args:
            entities: List of PII entity types to detect. If None, uses
                default entities focused on credentials and financial data.
            action: Action to take when PII is detected:
                - "redact": Replace PII with placeholder and pass
                - "block": Fail the guardrail check
                - "warn": Pass but include warning in result
            custom_patterns: Additional custom regex patterns to check.
                Keys are entity names, values are regex patterns.
            min_confidence: Minimum confidence threshold for detection.
        """
        self.entities = entities or self.DEFAULT_ENTITIES
        self.action = action
        self.min_confidence = min_confidence

        # Compile patterns for selected entities
        self._compiled_patterns: dict[str, tuple[re.Pattern[str], float]] = {}
        for entity in self.entities:
            entity_upper = entity.upper()
            if entity_upper in self.PII_PATTERNS:
                pattern, confidence = self.PII_PATTERNS[entity_upper]
                self._compiled_patterns[entity_upper] = (
                    re.compile(pattern, re.IGNORECASE if entity_upper != "AWS_ACCESS_KEY" else 0),
                    confidence,
                )
            else:
                logger.warning(f"Unknown PII entity type: {entity}")

        # Add custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self._compiled_patterns[name.upper()] = (
                    re.compile(pattern),
                    0.8,  # Default confidence for custom patterns
                )

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Check LLM output for PII leakage.

        Args:
            text: The LLM output text to check.
            context: Optional context containing expected_pii to allow.

        Returns:
            GuardrailResult with detection details and optional redacted text.
        """
        matches = self._detect_pii(text)

        # Filter by confidence threshold
        matches = [m for m in matches if m.confidence >= self.min_confidence]

        # Allow expected PII from context
        if context and "expected_pii" in context:
            expected = set(context["expected_pii"])
            matches = [m for m in matches if m.value not in expected]

        if not matches:
            return GuardrailResult(passed=True)

        # Group matches by entity type
        entity_counts: dict[str, int] = {}
        for match in matches:
            entity_counts[match.entity_type] = entity_counts.get(match.entity_type, 0) + 1

        # Determine severity based on PII types found
        severity = self._determine_severity(matches)

        # Build result details
        details: dict[str, Any] = {
            "entities_found": entity_counts,
            "match_count": len(matches),
            "matches": [
                {
                    "type": m.entity_type,
                    "start": m.start,
                    "end": m.end,
                    "confidence": m.confidence,
                }
                for m in matches
            ],
        }

        # Handle based on action
        if self.action == "redact":
            redacted_text = self._redact_text(text, matches)
            details["redacted_text"] = redacted_text
            details["original_length"] = len(text)
            return GuardrailResult(
                passed=True,
                reason=f"PII leakage detected and redacted: {', '.join(entity_counts.keys())}",
                severity=severity,
                details=details,
            )

        if self.action == "warn":
            return GuardrailResult(
                passed=True,
                reason=f"PII leakage detected (warning): {', '.join(entity_counts.keys())}",
                severity=severity,
                details=details,
            )

        # Default: block
        return GuardrailResult(
            passed=False,
            reason=f"PII leakage detected: {', '.join(entity_counts.keys())}",
            severity=severity,
            details=details,
        )

    async def check_async(
        self, text: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Async version of check (same implementation, no async operations)."""
        return self.check(text, context)

    def _detect_pii(self, text: str) -> list[PIILeakageMatch]:
        """Detect all PII entities in the text."""
        matches: list[PIILeakageMatch] = []

        for entity_type, (pattern, confidence) in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                value = match.group()

                # Additional validation for certain types
                adjusted_confidence = self._validate_match(entity_type, value, confidence)

                if adjusted_confidence > 0:
                    matches.append(
                        PIILeakageMatch(
                            entity_type=entity_type,
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            confidence=adjusted_confidence,
                        )
                    )

        # Sort by position
        matches.sort(key=lambda m: m.start)

        # Remove overlapping matches (keep higher confidence)
        return self._remove_overlaps(matches)

    def _validate_match(self, entity_type: str, value: str, base_confidence: float) -> float:
        """Apply additional validation to adjust confidence."""
        if entity_type == "CREDIT_CARD":
            # Validate with Luhn algorithm
            if not self._luhn_check(value.replace("-", "").replace(" ", "")):
                return 0.0

        elif entity_type == "SSN":
            # Validate SSN format
            cleaned = value.replace("-", "").replace(" ", "")
            if len(cleaned) != 9 or not cleaned.isdigit():
                return 0.0
            # Check for obviously fake SSNs
            if cleaned in ("123456789", "111111111", "000000000"):
                return base_confidence * 0.5

        elif entity_type == "AWS_SECRET_KEY":
            # Reduce confidence if it looks like base64 but isn't a typical secret
            if not any(c in value for c in ["/", "+", "="]):
                return base_confidence * 0.5

        elif entity_type == "BANK_ACCOUNT":
            # Very low confidence without context
            return base_confidence * 0.3

        return base_confidence

    def _luhn_check(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        if not number.isdigit():
            return False

        digits = [int(d) for d in number]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]

        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(divmod(d * 2, 10))

        return checksum % 10 == 0

    def _remove_overlaps(self, matches: list[PIILeakageMatch]) -> list[PIILeakageMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return matches

        result: list[PIILeakageMatch] = []
        for match in matches:
            # Check for overlap with existing matches
            overlaps = False
            for i, existing in enumerate(result):
                if match.start < existing.end and match.end > existing.start:
                    overlaps = True
                    # Keep the higher confidence match
                    if match.confidence > existing.confidence:
                        result[i] = match
                    break

            if not overlaps:
                result.append(match)

        return result

    def _determine_severity(
        self, matches: list[PIILeakageMatch]
    ) -> Literal["low", "medium", "high", "critical"]:
        """Determine severity based on PII types found."""
        critical_types = {"SSN", "CREDIT_CARD", "PRIVATE_KEY", "AWS_SECRET_KEY"}
        high_types = {"API_KEY", "AWS_ACCESS_KEY", "PASSWORD", "JWT", "BEARER_TOKEN"}
        medium_types = {"BANK_ACCOUNT", "ROUTING_NUMBER"}

        entity_types = {m.entity_type for m in matches}

        if entity_types & critical_types:
            return "critical"
        if entity_types & high_types:
            return "high"
        if entity_types & medium_types:
            return "medium"
        return "low"

    def _redact_text(self, text: str, matches: list[PIILeakageMatch]) -> str:
        """Redact PII from text, replacing with placeholders."""
        if not matches:
            return text

        # Sort by position (descending) to replace from end to start
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        result = text
        for match in sorted_matches:
            placeholder = f"[{match.entity_type}]"
            result = result[: match.start] + placeholder + result[match.end :]

        return result
