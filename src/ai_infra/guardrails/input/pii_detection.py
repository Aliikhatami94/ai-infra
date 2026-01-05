"""PII (Personally Identifiable Information) detection guardrail.

This module provides guardrails for detecting and handling PII in user input,
including email addresses, phone numbers, SSNs, credit card numbers, and more.

Example:
    >>> from ai_infra.guardrails.input import PIIDetection
    >>>
    >>> guard = PIIDetection(entities=["EMAIL", "PHONE"], action="redact")
    >>> result = guard.check("Email me at john@example.com")
    >>> print(result.details.get("redacted_text"))  # "Email me at [EMAIL]"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from ai_infra.guardrails.base import Guardrail, GuardrailResult

logger = logging.getLogger(__name__)


@dataclass
class PIIMatch:
    """Represents a detected PII entity."""

    entity_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0

    def __repr__(self) -> str:
        """Return a string representation."""
        masked = self.value[:2] + "..." if len(self.value) > 4 else "***"
        return f"PIIMatch({self.entity_type}, '{masked}', {self.start}:{self.end})"


class PIIDetection(Guardrail):
    """Guardrail for detecting and handling PII in text.

    Supports detection of common PII types using regex patterns:
    - EMAIL: Email addresses
    - PHONE: Phone numbers (US format)
    - SSN: Social Security Numbers
    - CREDIT_CARD: Credit card numbers
    - IP_ADDRESS: IPv4 and IPv6 addresses
    - DATE_OF_BIRTH: Birth dates
    - PASSPORT: Passport numbers (generic)

    Attributes:
        name: "pii_detection"
        entities: List of PII types to detect
        action: What to do when PII is found ("redact", "block", "warn")

    Example:
        >>> guard = PIIDetection(entities=["EMAIL", "SSN"], action="block")
        >>> result = guard.check("My SSN is 123-45-6789")
        >>> assert not result.passed
    """

    name = "pii_detection"

    # PII detection patterns
    PII_PATTERNS: dict[str, tuple[str, float]] = {
        "EMAIL": (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            0.99,
        ),
        "PHONE": (
            r"(?:\+?1[-.\s]?)?(?:\([2-9]\d{2}\)[-.\s]?|[2-9]\d{2}[-.\s]?)\d{3}[-.\s]?\d{4}",
            0.9,
        ),
        "SSN": (
            r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b",
            0.95,
        ),
        "CREDIT_CARD": (
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            0.95,
        ),
        "IP_ADDRESS": (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            0.99,
        ),
        "IPV6_ADDRESS": (
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
            0.99,
        ),
        "DATE_OF_BIRTH": (
            r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b",
            0.7,
        ),
        "PASSPORT": (
            r"\b[A-Z]{1,2}[0-9]{6,9}\b",
            0.6,
        ),
        "DRIVERS_LICENSE": (
            r"\b[A-Z]{1,2}\d{4,8}\b",
            0.5,
        ),
    }

    # Default entities if none specified
    DEFAULT_ENTITIES = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS"]

    def __init__(
        self,
        entities: list[str] | None = None,
        action: Literal["redact", "block", "warn"] = "block",
        custom_patterns: dict[str, str] | None = None,
        min_confidence: float = 0.5,
    ):
        """Initialize the PII detection guardrail.

        Args:
            entities: List of PII entity types to detect. If None, uses
                default entities (EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS).
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
                    re.compile(pattern, re.IGNORECASE),
                    confidence,
                )
            else:
                logger.warning(f"Unknown PII entity type: {entity}")

        # Add custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self._compiled_patterns[name.upper()] = (
                    re.compile(pattern, re.IGNORECASE),
                    0.8,  # Default confidence for custom patterns
                )

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Check text for PII entities.

        Args:
            text: The text to check for PII.
            context: Optional context (not used).

        Returns:
            GuardrailResult with detection details and optional redacted text.
        """
        matches = self._detect_pii(text)

        if not matches:
            return GuardrailResult(passed=True)

        # Filter by confidence threshold
        matches = [m for m in matches if m.confidence >= self.min_confidence]

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
                reason=f"PII detected and redacted: {', '.join(entity_counts.keys())}",
                severity=severity,
                details=details,
            )

        if self.action == "warn":
            return GuardrailResult(
                passed=True,
                reason=f"PII detected (warning): {', '.join(entity_counts.keys())}",
                severity=severity,
                details=details,
            )

        # Default: block
        return GuardrailResult(
            passed=False,
            reason=f"PII detected: {', '.join(entity_counts.keys())}",
            severity=severity,
            details=details,
        )

    def _detect_pii(self, text: str) -> list[PIIMatch]:
        """Detect all PII entities in the text."""
        matches: list[PIIMatch] = []

        for entity_type, (pattern, confidence) in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                # Additional validation for certain types
                value = match.group()
                adjusted_confidence = self._validate_match(entity_type, value, confidence)

                if adjusted_confidence > 0:
                    matches.append(
                        PIIMatch(
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
        """Validate and adjust confidence for specific entity types."""
        if entity_type == "SSN":
            # Additional SSN validation
            digits = re.sub(r"\D", "", value)
            if len(digits) != 9:
                return 0
            # Check for obviously invalid SSNs
            if digits.startswith(("000", "666")) or digits[3:5] == "00" or digits[5:] == "0000":
                return 0
            return base_confidence

        if entity_type == "CREDIT_CARD":
            # Luhn algorithm validation
            digits = re.sub(r"\D", "", value)
            if not self._luhn_check(digits):
                return base_confidence * 0.5  # Lower confidence if Luhn fails
            return base_confidence

        if entity_type == "PHONE":
            # Validate US phone number structure
            digits = re.sub(r"\D", "", value)
            if len(digits) < 10:
                return 0
            return base_confidence

        return base_confidence

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        try:
            digits = [int(d) for d in card_number]
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]

            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(divmod(d * 2, 10))

            return checksum % 10 == 0
        except ValueError:
            return False

    def _remove_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove overlapping matches, keeping higher confidence ones."""
        if not matches:
            return matches

        result: list[PIIMatch] = []
        for match in matches:
            # Check if this overlaps with any existing result
            overlaps = False
            for i, existing in enumerate(result):
                if not (match.end <= existing.start or match.start >= existing.end):
                    # Overlaps - keep higher confidence
                    if match.confidence > existing.confidence:
                        result[i] = match
                    overlaps = True
                    break
            if not overlaps:
                result.append(match)

        return result

    def _redact_text(self, text: str, matches: list[PIIMatch]) -> str:
        """Replace PII matches with redaction placeholders."""
        # Sort by position descending to replace from end
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        redacted = text
        for match in sorted_matches:
            placeholder = f"[{match.entity_type}]"
            redacted = redacted[: match.start] + placeholder + redacted[match.end :]

        return redacted

    def _determine_severity(
        self, matches: list[PIIMatch]
    ) -> Literal["low", "medium", "high", "critical"]:
        """Determine severity based on PII types found."""
        types = {m.entity_type for m in matches}

        # Critical: Financial or identity documents
        if types & {"SSN", "CREDIT_CARD", "PASSPORT"}:
            return "critical"

        # High: Contact information that could enable fraud
        if types & {"PHONE", "DATE_OF_BIRTH", "DRIVERS_LICENSE"}:
            return "high"

        # Medium: General PII
        if types & {"EMAIL", "IP_ADDRESS"}:
            return "medium"

        return "low"

    def redact(self, text: str) -> str:
        """Convenience method to redact PII from text.

        Args:
            text: Text containing potential PII.

        Returns:
            Text with PII replaced by placeholders.
        """
        matches = self._detect_pii(text)
        matches = [m for m in matches if m.confidence >= self.min_confidence]
        return self._redact_text(text, matches) if matches else text
