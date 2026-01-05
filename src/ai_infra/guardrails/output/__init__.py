"""Output guardrails for validating LLM outputs.

This submodule provides guardrails that validate LLM outputs before
they are returned to users.

Available guardrails:
- Toxicity: Detects toxic or harmful content using OpenAI Moderation API
- PIILeakage: Detects PII in outputs that shouldn't be there
- Hallucination: Detects potential hallucinations in RAG outputs

Example:
    >>> from ai_infra.guardrails.output import Toxicity, PIILeakage
    >>>
    >>> toxicity = Toxicity(threshold=0.7)
    >>> result = toxicity.check(llm_output)
    >>> if not result.passed:
    ...     print(f"Toxic content: {result.reason}")
"""

from __future__ import annotations

from ai_infra.guardrails.output.hallucination import Hallucination
from ai_infra.guardrails.output.pii_leakage import PIILeakage, PIILeakageMatch
from ai_infra.guardrails.output.toxicity import Toxicity

__all__ = [
    "Toxicity",
    "PIILeakage",
    "PIILeakageMatch",
    "Hallucination",
]
