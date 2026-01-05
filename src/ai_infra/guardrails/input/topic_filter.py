"""Topic filtering guardrail.

This module provides guardrails for filtering content based on topics,
using embedding-based semantic matching to detect when user input
discusses blocked topics.

Example:
    >>> from ai_infra.guardrails.input import TopicFilter
    >>>
    >>> guard = TopicFilter(
    ...     blocked_topics=["violence", "illegal activities"],
    ...     threshold=0.75,
    ... )
    >>> result = guard.check("How do I pick a lock?")
    >>> print(result.passed)  # False (if semantically similar to blocked topics)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from ai_infra.guardrails.base import Guardrail, GuardrailResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TopicFilter(Guardrail):
    """Guardrail for filtering content based on blocked topics.

    Uses embedding-based semantic similarity to detect when user input
    discusses or relates to blocked topics, even when exact keywords
    aren't used.

    Attributes:
        name: "topic_filter"
        blocked_topics: List of topics to filter
        threshold: Similarity threshold for blocking (0.0-1.0)
        method: Detection method ("embedding" or "keyword")

    Example:
        >>> guard = TopicFilter(
        ...     blocked_topics=["hacking", "weapons"],
        ...     threshold=0.8,
        ... )
        >>> result = guard.check("Tell me about cybersecurity vulnerabilities")
        >>> # May fail if semantically similar to "hacking"
    """

    name = "topic_filter"

    # Pre-defined topic categories with example phrases
    TOPIC_EXPANSIONS: dict[str, list[str]] = {
        "violence": [
            "how to hurt someone",
            "weapons for harming",
            "causing physical harm",
            "instructions for violence",
            "attacking people",
        ],
        "illegal_activities": [
            "how to break the law",
            "illegal methods",
            "criminal activity",
            "avoiding law enforcement",
            "committing crimes",
        ],
        "hacking": [
            "how to hack",
            "breaking into systems",
            "unauthorized access",
            "exploiting vulnerabilities",
            "bypassing security",
        ],
        "self_harm": [
            "hurting myself",
            "self-inflicted harm",
            "methods of self-harm",
            "suicidal thoughts",
            "ending my life",
        ],
        "hate_speech": [
            "derogatory language",
            "discriminatory content",
            "racial slurs",
            "hateful rhetoric",
            "targeting groups",
        ],
        "explicit_content": [
            "adult content",
            "sexually explicit",
            "pornographic material",
            "nsfw content",
            "sexual acts",
        ],
        "fraud": [
            "scamming people",
            "identity theft",
            "financial fraud",
            "fake documents",
            "impersonation",
        ],
        "drugs": [
            "illegal substances",
            "drug manufacturing",
            "how to get drugs",
            "drug dealing",
            "controlled substances",
        ],
    }

    # Keyword fallback patterns for each topic
    TOPIC_KEYWORDS: dict[str, list[str]] = {
        "violence": [
            "kill",
            "murder",
            "attack",
            "hurt",
            "harm",
            "weapon",
            "assault",
            "fight",
            "punch",
            "stab",
            "shoot",
        ],
        "illegal_activities": [
            "illegal",
            "crime",
            "steal",
            "rob",
            "fraud",
            "forge",
            "smuggle",
            "trafficking",
            "launder",
        ],
        "hacking": [
            "hack",
            "exploit",
            "crack",
            "bypass",
            "breach",
            "backdoor",
            "malware",
            "virus",
            "trojan",
            "phishing",
            "ddos",
        ],
        "self_harm": [
            "suicide",
            "self-harm",
            "cut myself",
            "end my life",
            "hurt myself",
            "hurting myself",
            "overdose",
            "self harm",
        ],
        "hate_speech": [
            "hate",
            "slur",
            "racist",
            "sexist",
            "discriminate",
            "bigot",
            "supremacy",
        ],
        "explicit_content": [
            "porn",
            "nude",
            "nsfw",
            "xxx",
            "explicit",
            "erotic",
        ],
        "fraud": [
            "scam",
            "fraud",
            "fake id",
            "counterfeit",
            "phishing",
            "impersonate",
            "identity theft",
        ],
        "drugs": [
            "cocaine",
            "heroin",
            "meth",
            "drug deal",
            "buy drugs",
            "make drugs",
            "synthesize",
        ],
    }

    def __init__(
        self,
        blocked_topics: list[str],
        threshold: float = 0.75,
        method: Literal["embedding", "keyword"] = "embedding",
        model: str | None = None,
        custom_topic_phrases: dict[str, list[str]] | None = None,
    ):
        """Initialize the topic filter guardrail.

        Args:
            blocked_topics: List of topics to block. Can be predefined
                topics (violence, hacking, etc.) or custom topic names
                (when using custom_topic_phrases).
            threshold: Similarity threshold for blocking (0.0-1.0).
                Higher values require closer semantic match.
            method: Detection method:
                - "embedding": Uses embedding similarity (more accurate)
                - "keyword": Uses keyword matching (faster, no dependencies)
            model: Embedding model name for "embedding" method.
            custom_topic_phrases: Additional topic definitions with
                example phrases for semantic matching.
        """
        self.blocked_topics = [t.lower().replace(" ", "_") for t in blocked_topics]
        self.threshold = threshold
        self.method = method
        self.model = model

        # Build topic phrase mappings
        self._topic_phrases: dict[str, list[str]] = {}
        self._topic_keywords: dict[str, list[str]] = {}

        for topic in self.blocked_topics:
            # Add predefined expansions
            if topic in self.TOPIC_EXPANSIONS:
                self._topic_phrases[topic] = self.TOPIC_EXPANSIONS[topic].copy()
            else:
                # Use topic name as basic phrase
                self._topic_phrases[topic] = [topic.replace("_", " ")]

            # Add predefined keywords
            if topic in self.TOPIC_KEYWORDS:
                self._topic_keywords[topic] = self.TOPIC_KEYWORDS[topic].copy()
            else:
                # Use topic words as keywords
                self._topic_keywords[topic] = topic.replace("_", " ").split()

        # Add custom topic phrases
        if custom_topic_phrases:
            for topic, phrases in custom_topic_phrases.items():
                topic_key = topic.lower().replace(" ", "_")
                if topic_key in self._topic_phrases:
                    self._topic_phrases[topic_key].extend(phrases)
                else:
                    self._topic_phrases[topic_key] = phrases
                    self._topic_keywords[topic_key] = [
                        word for phrase in phrases for word in phrase.split()
                    ]

        # Cache for embeddings
        self._embedder: Any | None = None
        self._topic_embeddings: dict[str, Any] | None = None

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Check if text discusses any blocked topics.

        Args:
            text: The text to check for blocked topics.
            context: Optional context (not used).

        Returns:
            GuardrailResult indicating whether the check passed.
        """
        if self.method == "embedding":
            return self._check_embedding(text)
        return self._check_keyword(text)

    def _check_keyword(self, text: str) -> GuardrailResult:
        """Check for blocked topics using keyword matching."""
        text_lower = text.lower()
        matched_topics: dict[str, list[str]] = {}

        for topic, keywords in self._topic_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                matched_topics[topic] = matches

        if not matched_topics:
            return GuardrailResult(passed=True)

        # Determine severity based on number of matches
        total_matches = sum(len(m) for m in matched_topics.values())
        severity = self._get_severity(len(matched_topics), total_matches)

        return GuardrailResult(
            passed=False,
            reason=f"Blocked topics detected: {', '.join(matched_topics.keys())}",
            severity=severity,
            details={
                "matched_topics": matched_topics,
                "total_keyword_matches": total_matches,
            },
        )

    def _check_embedding(self, text: str) -> GuardrailResult:
        """Check for blocked topics using embedding similarity."""
        try:
            # Initialize embedder if needed
            if self._embedder is None:
                self._initialize_embedder()

            if self._embedder is None:
                logger.warning("Embedder not available, falling back to keyword matching")
                return self._check_keyword(text)

            # Get text embedding
            text_embedding = self._embedder.embed(text)

            # Compare with topic embeddings
            similarities: dict[str, float] = {}
            for topic, topic_embedding in (self._topic_embeddings or {}).items():
                similarity = self._cosine_similarity(text_embedding, topic_embedding)
                if similarity >= self.threshold:
                    similarities[topic] = similarity

            if not similarities:
                return GuardrailResult(passed=True)

            # Get highest similarity
            max_topic = max(similarities, key=lambda t: similarities[t])
            max_similarity = similarities[max_topic]

            severity = (
                "critical" if max_similarity > 0.9 else "high" if max_similarity > 0.8 else "medium"
            )

            return GuardrailResult(
                passed=False,
                reason=f"Blocked topic detected: {max_topic} (similarity: {max_similarity:.2f})",
                severity=severity,
                details={
                    "matched_topics": similarities,
                    "highest_match": max_topic,
                    "highest_similarity": max_similarity,
                },
            )

        except Exception as e:
            logger.error(f"Embedding-based check failed: {e}, falling back to keyword matching")
            return self._check_keyword(text)

    def _initialize_embedder(self) -> None:
        """Initialize the embedding model and compute topic embeddings."""
        try:
            from ai_infra import Embedder

            self._embedder = Embedder(model=self.model) if self.model else Embedder()

            # Compute embeddings for topic phrases
            self._topic_embeddings = {}
            for topic, phrases in self._topic_phrases.items():
                # Combine phrases for topic embedding
                combined_text = " ".join(phrases)
                self._topic_embeddings[topic] = self._embedder.embed(combined_text)

        except ImportError:
            logger.warning("Embedder not available, will use keyword matching")
            self._embedder = None
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            self._embedder = None

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np

            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except ImportError:
            # Fallback without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

    def _get_severity(
        self, topic_count: int, match_count: int
    ) -> Literal["low", "medium", "high", "critical"]:
        """Determine severity based on match counts."""
        if topic_count >= 3 or match_count >= 5:
            return "critical"
        if topic_count >= 2 or match_count >= 3:
            return "high"
        if match_count >= 2:
            return "medium"
        return "low"

    def add_topic(self, topic: str, phrases: list[str], keywords: list[str] | None = None) -> None:
        """Add a custom topic to the filter.

        Args:
            topic: Topic name.
            phrases: Example phrases for semantic matching.
            keywords: Optional keywords for keyword matching.
        """
        topic_key = topic.lower().replace(" ", "_")
        self.blocked_topics.append(topic_key)
        self._topic_phrases[topic_key] = phrases
        self._topic_keywords[topic_key] = keywords or [
            word for phrase in phrases for word in phrase.split()
        ]

        # Clear cached embeddings
        self._topic_embeddings = None
