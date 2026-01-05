"""Toxicity detection guardrail for LLM outputs.

This module provides guardrails for detecting toxic or harmful content
in LLM outputs, including hate speech, harassment, violence, and sexual content.

Detection methods:
- OpenAI: Uses OpenAI's free Moderation API (recommended)
- Heuristic: Fast pattern matching for obvious toxic content

Example:
    >>> from ai_infra.guardrails.output import Toxicity
    >>>
    >>> guard = Toxicity(threshold=0.7)
    >>> result = guard.check(llm_output)
    >>> if not result.passed:
    ...     print(f"Toxic content detected: {result.reason}")
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from ai_infra.guardrails.base import Guardrail, GuardrailResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Toxicity(Guardrail):
    """Guardrail for detecting toxic or harmful content in LLM outputs.

    Uses content moderation to detect various categories of harmful content
    including hate speech, harassment, self-harm, sexual content, and violence.

    Attributes:
        name: "toxicity"
        threshold: Score threshold above which content is flagged (0.0-1.0)
        method: Detection method ("openai", "heuristic")
        categories: Categories of content to check for

    Example:
        >>> guard = Toxicity(threshold=0.7)
        >>> result = guard.check("Some potentially harmful text")
        >>> if not result.passed:
        ...     print(f"Blocked: {result.reason}")
    """

    name = "toxicity"

    # OpenAI moderation categories
    OPENAI_CATEGORIES = [
        "hate",
        "hate/threatening",
        "harassment",
        "harassment/threatening",
        "self-harm",
        "self-harm/intent",
        "self-harm/instructions",
        "sexual",
        "sexual/minors",
        "violence",
        "violence/graphic",
    ]

    # Simplified category mapping for user-facing API
    CATEGORY_GROUPS = {
        "hate": ["hate", "hate/threatening"],
        "harassment": ["harassment", "harassment/threatening"],
        "self_harm": ["self-harm", "self-harm/intent", "self-harm/instructions"],
        "sexual": ["sexual", "sexual/minors"],
        "violence": ["violence", "violence/graphic"],
    }

    # Heuristic patterns for common toxic content
    TOXIC_PATTERNS = [
        # Slurs and hate speech (simplified, non-exhaustive)
        (r"\b(?:kill|murder|attack)\s+(?:all|every|those)\s+\w+", "violence"),
        (r"\b(?:deserve[sd]?\s+to\s+die)\b", "violence"),
        (r"\bhow\s+to\s+(?:kill|hurt|harm)\b", "violence"),
        # Self-harm indicators
        (r"\b(?:cut|hurt)\s+(?:myself|yourself)\b", "self_harm"),
        (r"\bsuicide\s+(?:method|how|way)\b", "self_harm"),
        # Harassment patterns
        (r"\byou\s+(?:should|deserve)\s+(?:die|suffer)\b", "harassment"),
        (r"\b(?:threat|threaten)\b.*\b(?:you|your|family)\b", "harassment"),
    ]

    def __init__(
        self,
        threshold: float = 0.7,
        categories: list[str] | None = None,
        method: Literal["openai", "heuristic"] = "openai",
        api_key: str | None = None,
    ):
        """Initialize the toxicity detection guardrail.

        Args:
            threshold: Score threshold (0.0-1.0) above which content is
                flagged as toxic. Lower values are more sensitive.
            categories: Categories to check. Options: "hate", "harassment",
                "self_harm", "sexual", "violence". If None, checks all.
            method: Detection method:
                - "openai": Uses OpenAI Moderation API (free, accurate)
                - "heuristic": Pattern matching (fast, less accurate)
            api_key: OpenAI API key for "openai" method. If None, uses
                OPENAI_API_KEY environment variable.
        """
        self.threshold = threshold
        self.categories = categories or list(self.CATEGORY_GROUPS.keys())
        self.method = method
        self._api_key = api_key

        # Compile heuristic patterns
        self._compiled_patterns: list[tuple[re.Pattern[str], str]] = []
        for pattern, category in self.TOXIC_PATTERNS:
            if category in self.categories:
                self._compiled_patterns.append((re.compile(pattern, re.IGNORECASE), category))

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Check text for toxic content.

        Args:
            text: The LLM output text to check.
            context: Optional context (not used).

        Returns:
            GuardrailResult indicating whether the check passed.
        """
        if self.method == "openai":
            return self._check_openai(text)
        return self._check_heuristic(text)

    async def check_async(
        self, text: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Async version of check for OpenAI method.

        Args:
            text: The LLM output text to check.
            context: Optional context (not used).

        Returns:
            GuardrailResult indicating whether the check passed.
        """
        if self.method == "openai":
            return await self._check_openai_async(text)
        return self._check_heuristic(text)

    def _check_heuristic(self, text: str) -> GuardrailResult:
        """Check for toxic content using pattern matching."""
        detections: list[dict[str, Any]] = []

        for pattern, category in self._compiled_patterns:
            for match in pattern.finditer(text):
                detections.append(
                    {
                        "category": category,
                        "matched_text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        if not detections:
            return GuardrailResult(passed=True)

        # Determine severity based on categories found
        categories_found = list({d["category"] for d in detections})
        severity = self._determine_severity(categories_found)

        return GuardrailResult(
            passed=False,
            reason=f"Toxic content detected: {', '.join(categories_found)}",
            severity=severity,
            details={
                "method": "heuristic",
                "categories_flagged": categories_found,
                "detections": detections,
            },
        )

    def _check_openai(self, text: str) -> GuardrailResult:
        """Check for toxic content using OpenAI Moderation API."""
        try:
            import httpx
        except ImportError as err:
            msg = "httpx is required for OpenAI moderation. Install with: pip install httpx"
            raise ImportError(msg) from err

        api_key = self._get_api_key()
        if not api_key:
            logger.warning("No OpenAI API key found, falling back to heuristic method")
            return self._check_heuristic(text)

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    "https://api.openai.com/v1/moderations",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": text},
                )
                response.raise_for_status()
                return self._parse_openai_response(response.json())
        except httpx.HTTPStatusError as e:
            logger.warning(f"OpenAI moderation API error: {e}, falling back to heuristic")
            return self._check_heuristic(text)
        except httpx.TimeoutException:
            logger.warning("OpenAI moderation API timeout, falling back to heuristic")
            return self._check_heuristic(text)

    async def _check_openai_async(self, text: str) -> GuardrailResult:
        """Async check for toxic content using OpenAI Moderation API."""
        try:
            import httpx
        except ImportError as err:
            msg = "httpx is required for OpenAI moderation. Install with: pip install httpx"
            raise ImportError(msg) from err

        api_key = self._get_api_key()
        if not api_key:
            logger.warning("No OpenAI API key found, falling back to heuristic method")
            return self._check_heuristic(text)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/moderations",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": text},
                )
                response.raise_for_status()
                return self._parse_openai_response(response.json())
        except httpx.HTTPStatusError as e:
            logger.warning(f"OpenAI moderation API error: {e}, falling back to heuristic")
            return self._check_heuristic(text)
        except httpx.TimeoutException:
            logger.warning("OpenAI moderation API timeout, falling back to heuristic")
            return self._check_heuristic(text)

    def _parse_openai_response(self, response: dict[str, Any]) -> GuardrailResult:
        """Parse OpenAI moderation API response."""
        results = response.get("results", [])
        if not results:
            return GuardrailResult(passed=True)

        result = results[0]
        flagged = result.get("flagged", False)
        categories = result.get("categories", {})
        category_scores = result.get("category_scores", {})

        # Check only the categories we care about
        flagged_categories: list[str] = []
        category_details: dict[str, float] = {}

        for group_name, openai_cats in self.CATEGORY_GROUPS.items():
            if group_name not in self.categories:
                continue

            for cat in openai_cats:
                score = category_scores.get(cat, 0.0)
                if score >= self.threshold or categories.get(cat, False):
                    flagged_categories.append(group_name)
                    category_details[cat] = score
                    break

        # Remove duplicates while preserving order
        flagged_categories = list(dict.fromkeys(flagged_categories))

        if not flagged_categories:
            return GuardrailResult(
                passed=True,
                details={
                    "method": "openai",
                    "category_scores": category_scores,
                },
            )

        severity = self._determine_severity(flagged_categories)

        return GuardrailResult(
            passed=False,
            reason=f"Toxic content detected: {', '.join(flagged_categories)}",
            severity=severity,
            details={
                "method": "openai",
                "categories_flagged": flagged_categories,
                "category_scores": category_details,
                "openai_flagged": flagged,
            },
        )

    def _determine_severity(
        self, categories: list[str]
    ) -> Literal["low", "medium", "high", "critical"]:
        """Determine severity based on flagged categories."""
        critical_categories = {"self_harm", "violence"}
        high_categories = {"hate", "harassment"}

        if any(cat in critical_categories for cat in categories):
            return "critical"
        if any(cat in high_categories for cat in categories):
            return "high"
        return "medium"

    def _get_api_key(self) -> str | None:
        """Get OpenAI API key from config or environment."""
        if self._api_key:
            return self._api_key

        import os

        return os.environ.get("OPENAI_API_KEY")
