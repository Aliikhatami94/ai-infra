"""Hallucination detection guardrail for RAG applications.

This module provides guardrails for detecting when LLM outputs contain
information that is not grounded in the provided source documents,
helping to ensure factual accuracy in RAG (Retrieval-Augmented Generation)
applications.

Detection methods:
- NLI (Natural Language Inference): Uses entailment classification
- Embedding: Uses semantic similarity between output and sources
- LLM: Uses an LLM to verify factual grounding

Example:
    >>> from ai_infra.guardrails.output import Hallucination
    >>>
    >>> guard = Hallucination(method="nli", threshold=0.8)
    >>> result = guard.check(
    ...     llm_output,
    ...     context={"sources": retrieved_documents}
    ... )
    >>> if not result.passed:
    ...     print(f"Potential hallucination: {result.reason}")
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from ai_infra.guardrails.base import Guardrail, GuardrailResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Hallucination(Guardrail):
    """Guardrail for detecting hallucinations in RAG outputs.

    Verifies that LLM outputs are grounded in the provided source documents.
    This helps prevent the model from generating factually incorrect information
    that is not supported by the retrieved context.

    Attributes:
        name: "hallucination"
        method: Detection method ("nli", "embedding", "llm", "heuristic")
        threshold: Confidence threshold for grounding (0.0-1.0)

    Example:
        >>> guard = Hallucination(method="embedding", threshold=0.7)
        >>> result = guard.check(
        ...     "Paris is the capital of France",
        ...     context={"sources": ["France's capital city is Paris..."]}
        ... )
        >>> assert result.passed  # Output is grounded in sources
    """

    name = "hallucination"

    # Claims that typically need verification
    CLAIM_INDICATORS = [
        r"\b(?:is|are|was|were|has|have|had)\b",
        r"\b(?:according to|based on|research shows|studies show)\b",
        r"\b(?:approximately|about|around|exactly|precisely)\s+\d+",
        r"\b(?:in\s+)?\d{4}\b",  # Years
        r"\b(?:percent|%)\b",
        r"\b(?:million|billion|trillion)\b",
    ]

    # Hedging language (less likely to be hallucination)
    HEDGING_PATTERNS = [
        r"\b(?:might|may|could|possibly|perhaps|probably)\b",
        r"\b(?:I think|I believe|it seems|it appears)\b",
        r"\b(?:not sure|uncertain|unclear)\b",
    ]

    def __init__(
        self,
        method: Literal["nli", "embedding", "llm", "heuristic"] = "embedding",
        threshold: float = 0.7,
        model: str | None = None,
        embedding_model: str | None = None,
        check_claims_only: bool = True,
    ):
        """Initialize the hallucination detection guardrail.

        Args:
            method: Detection method:
                - "nli": Natural Language Inference entailment check
                - "embedding": Semantic similarity with sources
                - "llm": Use LLM to verify grounding
                - "heuristic": Basic pattern-based checks
            threshold: Confidence threshold (0.0-1.0). Higher values require
                stronger grounding in sources.
            model: LLM model to use for "llm" method.
            embedding_model: Embedding model for "embedding" method.
            check_claims_only: If True, only check sentences that appear
                to make factual claims.
        """
        self.method = method
        self.threshold = threshold
        self.model = model
        self.embedding_model = embedding_model or "text-embedding-3-small"
        self.check_claims_only = check_claims_only

        # Compile claim detection patterns
        self._claim_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLAIM_INDICATORS]
        self._hedging_patterns = [re.compile(p, re.IGNORECASE) for p in self.HEDGING_PATTERNS]

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Check LLM output for hallucinations.

        Args:
            text: The LLM output text to check.
            context: Context containing "sources" - list of source documents
                that the output should be grounded in.

        Returns:
            GuardrailResult indicating whether the output is grounded.

        Raises:
            ValueError: If context is missing required "sources" key.
        """
        if not context or "sources" not in context:
            # Without sources, we can only do heuristic checking
            if self.method == "heuristic":
                return self._check_heuristic(text)
            return GuardrailResult(
                passed=True,
                reason="No sources provided for grounding check",
                details={"skipped": True, "reason": "no_sources"},
            )

        sources = context["sources"]
        if not sources:
            return GuardrailResult(
                passed=True,
                reason="Empty sources list",
                details={"skipped": True, "reason": "empty_sources"},
            )

        # Normalize sources to list of strings
        source_texts = self._normalize_sources(sources)

        if self.method == "embedding":
            return self._check_embedding(text, source_texts)
        elif self.method == "nli":
            return self._check_nli(text, source_texts)
        elif self.method == "llm":
            return self._check_llm(text, source_texts)
        else:
            return self._check_heuristic(text, source_texts)

    async def check_async(
        self, text: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Async version of check."""
        if not context or "sources" not in context:
            if self.method == "heuristic":
                return self._check_heuristic(text)
            return GuardrailResult(
                passed=True,
                reason="No sources provided for grounding check",
                details={"skipped": True, "reason": "no_sources"},
            )

        sources = context["sources"]
        if not sources:
            return GuardrailResult(
                passed=True,
                reason="Empty sources list",
                details={"skipped": True, "reason": "empty_sources"},
            )

        source_texts = self._normalize_sources(sources)

        if self.method == "embedding":
            return await self._check_embedding_async(text, source_texts)
        elif self.method == "nli":
            return await self._check_nli_async(text, source_texts)
        elif self.method == "llm":
            return await self._check_llm_async(text, source_texts)
        else:
            return self._check_heuristic(text, source_texts)

    def _normalize_sources(self, sources: Any) -> list[str]:
        """Normalize sources to a list of strings."""
        if isinstance(sources, str):
            return [sources]
        if isinstance(sources, list):
            result = []
            for s in sources:
                if isinstance(s, str):
                    result.append(s)
                elif isinstance(s, dict):
                    # Handle common document formats
                    content = s.get("content") or s.get("text") or s.get("page_content")
                    if content:
                        result.append(str(content))
                else:
                    result.append(str(s))
            return result
        return [str(sources)]

    def _extract_claims(self, text: str) -> list[str]:
        """Extract sentences that appear to make factual claims."""
        # Split into sentences (simple approach)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        if not self.check_claims_only:
            return sentences

        claims = []
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence) < 20:
                continue

            # Skip if it's hedged language
            if any(p.search(sentence) for p in self._hedging_patterns):
                continue

            # Include if it looks like a factual claim
            if any(p.search(sentence) for p in self._claim_patterns):
                claims.append(sentence)

        return claims if claims else sentences[:3]  # Fallback to first 3 sentences

    def _check_heuristic(self, text: str, sources: list[str] | None = None) -> GuardrailResult:
        """Basic heuristic check for potential hallucinations."""
        claims = self._extract_claims(text)

        if not claims:
            return GuardrailResult(passed=True)

        # Check for warning signs of hallucination
        warnings: list[str] = []

        # Check for overly specific claims (numbers, dates)
        specific_claim_pattern = re.compile(
            r"\b(?:exactly|precisely)\s+\d+|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
        )
        for claim in claims:
            if specific_claim_pattern.search(claim):
                warnings.append(f"Overly specific claim: {claim[:50]}...")

        # If sources provided, check for claims not in sources
        if sources:
            source_text = " ".join(sources).lower()
            for claim in claims:
                # Extract key terms from claim
                terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim)
                for term in terms:
                    if len(term) > 3 and term.lower() not in source_text:
                        warnings.append(f"Term not in sources: {term}")
                        break

        if not warnings:
            return GuardrailResult(passed=True, details={"method": "heuristic"})

        # Determine severity based on number of warnings
        if len(warnings) >= 3:
            severity: Literal["low", "medium", "high", "critical"] = "high"
        elif len(warnings) >= 2:
            severity = "medium"
        else:
            severity = "low"

        return GuardrailResult(
            passed=False,
            reason=f"Potential hallucination detected: {len(warnings)} warning(s)",
            severity=severity,
            details={
                "method": "heuristic",
                "warnings": warnings,
                "claims_checked": len(claims),
            },
        )

    def _check_embedding(self, text: str, sources: list[str]) -> GuardrailResult:
        """Check grounding using embedding similarity."""
        try:
            from ai_infra import Embeddings
        except ImportError:
            logger.warning("Embeddings not available, falling back to heuristic")
            return self._check_heuristic(text, sources)

        try:
            embeddings = Embeddings(model=self.embedding_model)
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            return self._check_heuristic(text, sources)

        claims = self._extract_claims(text)
        if not claims:
            return GuardrailResult(passed=True, details={"method": "embedding"})

        try:
            # Get embeddings for claims and sources
            claim_embeddings = embeddings.embed_batch(claims)
            source_embeddings = embeddings.embed_batch(sources)

            # Calculate max similarity for each claim
            ungrounded_claims = []
            claim_scores = []

            for i, claim in enumerate(claims):
                max_similarity = 0.0
                for source_emb in source_embeddings:
                    similarity = self._cosine_similarity(claim_embeddings[i], source_emb)
                    max_similarity = max(max_similarity, similarity)

                claim_scores.append(max_similarity)
                if max_similarity < self.threshold:
                    ungrounded_claims.append(
                        {
                            "claim": claim,
                            "max_similarity": max_similarity,
                        }
                    )

            avg_similarity = sum(claim_scores) / len(claim_scores) if claim_scores else 0

            if not ungrounded_claims:
                return GuardrailResult(
                    passed=True,
                    details={
                        "method": "embedding",
                        "avg_similarity": avg_similarity,
                        "claims_checked": len(claims),
                    },
                )

            # Determine severity
            ungrounded_ratio = len(ungrounded_claims) / len(claims)
            if ungrounded_ratio > 0.5:
                severity: Literal["low", "medium", "high", "critical"] = "high"
            elif ungrounded_ratio > 0.25:
                severity = "medium"
            else:
                severity = "low"

            return GuardrailResult(
                passed=False,
                reason=f"Potential hallucination: {len(ungrounded_claims)}/{len(claims)} claims not grounded",
                severity=severity,
                details={
                    "method": "embedding",
                    "avg_similarity": avg_similarity,
                    "ungrounded_claims": ungrounded_claims,
                    "claims_checked": len(claims),
                    "threshold": self.threshold,
                },
            )

        except Exception as e:
            logger.warning(f"Embedding check failed: {e}, falling back to heuristic")
            return self._check_heuristic(text, sources)

    async def _check_embedding_async(self, text: str, sources: list[str]) -> GuardrailResult:
        """Async version of embedding check."""
        # For now, use sync version (embeddings are typically fast)
        return self._check_embedding(text, sources)

    def _check_nli(self, text: str, sources: list[str]) -> GuardrailResult:
        """Check grounding using NLI (Natural Language Inference)."""
        # NLI requires a specialized model - fall back to embedding for now
        logger.info("NLI method not yet implemented, using embedding fallback")
        return self._check_embedding(text, sources)

    async def _check_nli_async(self, text: str, sources: list[str]) -> GuardrailResult:
        """Async version of NLI check."""
        return self._check_nli(text, sources)

    def _check_llm(self, text: str, sources: list[str]) -> GuardrailResult:
        """Check grounding using an LLM."""
        try:
            from ai_infra import LLM
        except ImportError:
            logger.warning("LLM not available, falling back to heuristic")
            return self._check_heuristic(text, sources)

        try:
            llm = LLM()
            model_name = self.model or "gpt-4o-mini"

            # Prepare verification prompt
            sources_text = "\n\n---\n\n".join(sources[:5])  # Limit to 5 sources
            prompt = f"""Analyze if the following response is factually grounded in the provided sources.

SOURCES:
{sources_text}

RESPONSE TO VERIFY:
{text}

For each factual claim in the response, determine if it is:
1. SUPPORTED: Directly stated or clearly implied by the sources
2. NOT SUPPORTED: Not mentioned in the sources or contradicts them
3. PARTIALLY SUPPORTED: Some aspects supported, others not

Respond in JSON format:
{{
    "overall_grounded": true/false,
    "confidence": 0.0-1.0,
    "claims": [
        {{"claim": "...", "status": "SUPPORTED/NOT SUPPORTED/PARTIALLY SUPPORTED", "reason": "..."}}
    ]
}}"""

            response = llm.chat(prompt, model_name=model_name)

            # Parse response
            import json

            try:
                # Extract JSON from response
                json_match = re.search(r"\{[\s\S]*\}", response.content)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

                grounded = result_data.get("overall_grounded", True)
                confidence = result_data.get("confidence", 0.5)
                claims_analysis = result_data.get("claims", [])

                if grounded and confidence >= self.threshold:
                    return GuardrailResult(
                        passed=True,
                        details={
                            "method": "llm",
                            "confidence": confidence,
                            "claims_analysis": claims_analysis,
                        },
                    )

                # Count unsupported claims
                unsupported = [
                    c
                    for c in claims_analysis
                    if c.get("status") in ("NOT SUPPORTED", "PARTIALLY SUPPORTED")
                ]

                severity: Literal["low", "medium", "high", "critical"]
                if len(unsupported) > len(claims_analysis) / 2:
                    severity = "high"
                elif confidence < 0.5:
                    severity = "medium"
                else:
                    severity = "low"

                return GuardrailResult(
                    passed=False,
                    reason=f"Potential hallucination: {len(unsupported)} unsupported claims",
                    severity=severity,
                    details={
                        "method": "llm",
                        "confidence": confidence,
                        "unsupported_claims": unsupported,
                        "claims_analysis": claims_analysis,
                    },
                )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response: {e}")
                return self._check_heuristic(text, sources)

        except Exception as e:
            logger.warning(f"LLM check failed: {e}, falling back to heuristic")
            return self._check_heuristic(text, sources)

    async def _check_llm_async(self, text: str, sources: list[str]) -> GuardrailResult:
        """Async version of LLM check."""
        # For now, use sync version
        return self._check_llm(text, sources)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
