"""Custom evaluators for ai-infra.

This module provides ai-infra specific evaluators that integrate with
ai_infra.Embeddings, OpenTelemetry spans, and LLM-based judging.

These evaluators extend pydantic-evals' Evaluator base class and are designed
to work seamlessly with the ai-infra ecosystem.

Example:
    >>> from ai_infra.eval.evaluators import SemanticSimilarity, ToolUsageEvaluator
    >>> from pydantic_evals import Case, Dataset
    >>>
    >>> dataset = Dataset(
    ...     cases=[Case(inputs="test", expected_output="expected")],
    ...     evaluators=[
    ...         SemanticSimilarity(threshold=0.8),
    ...         ToolUsageEvaluator(expected_tools=["get_weather"]),
    ...     ],
    ... )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)

if TYPE_CHECKING:
    from ai_infra.embeddings import Embeddings


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


@dataclass
class SemanticSimilarity(Evaluator[str, str]):
    """Evaluate semantic similarity between output and expected output.

    Uses ai_infra.Embeddings to compute cosine similarity between the
    output and expected_output embeddings.

    Args:
        provider: Embedding provider (openai, google, voyage, cohere, huggingface).
            If None, auto-detects from environment.
        model: Embedding model name. Uses provider default if not specified.
        threshold: Minimum similarity score to pass (0.0-1.0). Default: 0.8.
        embeddings: Pre-configured Embeddings instance. If provided,
            `provider` and `model` are ignored.

    Example:
        >>> from ai_infra.eval.evaluators import SemanticSimilarity
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(
        ...             inputs="What is the capital of France?",
        ...             expected_output="Paris is the capital",
        ...         ),
        ...     ],
        ...     evaluators=[SemanticSimilarity(threshold=0.7)],
        ... )

    Returns:
        EvaluationReason with:
        - value: float (similarity score 0.0-1.0)
        - reason: Explanation of the score and pass/fail
    """

    provider: str | None = None
    model: str | None = None
    threshold: float = 0.8
    embeddings: Embeddings | None = field(default=None, repr=False)

    # Cached embeddings instance
    _embeddings_instance: Embeddings | None = field(default=None, init=False, repr=False)

    def _get_embeddings(self) -> Embeddings:
        """Get or create the Embeddings instance."""
        if self._embeddings_instance is not None:
            return self._embeddings_instance

        if self.embeddings is not None:
            self._embeddings_instance = self.embeddings
        else:
            # Lazy import to avoid circular dependencies
            from ai_infra.embeddings import Embeddings

            self._embeddings_instance = Embeddings(
                provider=self.provider,
                model=self.model,
            )

        return self._embeddings_instance

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> EvaluatorOutput:
        """Evaluate semantic similarity between output and expected output."""
        if ctx.expected_output is None:
            return EvaluationReason(
                value=True,
                reason="Skipped: no expected output provided",
            )

        if not ctx.output or not ctx.expected_output:
            return EvaluationReason(
                value=0.0,
                reason="Empty output or expected output",
            )

        try:
            embeddings = self._get_embeddings()

            # Embed both texts
            output_embedding = embeddings.embed(str(ctx.output))
            expected_embedding = embeddings.embed(str(ctx.expected_output))

            # Calculate cosine similarity
            similarity = _cosine_similarity(output_embedding, expected_embedding)

            passed = similarity >= self.threshold
            return EvaluationReason(
                value=similarity,
                reason=(
                    f"Similarity {similarity:.3f} >= threshold {self.threshold}"
                    if passed
                    else f"Similarity {similarity:.3f} < threshold {self.threshold}"
                ),
            )
        except Exception as e:
            return EvaluationReason(
                value=0.0,
                reason=f"Error computing similarity: {e}",
            )


@dataclass
class ToolUsageEvaluator(Evaluator[Any, Any]):
    """Evaluate that an agent called expected tools.

    Uses span-based evaluation with OpenTelemetry to check which tools
    were called during agent execution.

    Args:
        expected_tools: List of tool names that should have been called.
        forbidden_tools: List of tool names that should NOT have been called.
        require_all: If True, all expected_tools must be called. If False,
            at least one must be called. Default: True.
        check_order: If True, expected_tools must be called in order.
            Default: False.

    Example:
        >>> from ai_infra.eval.evaluators import ToolUsageEvaluator
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> dataset = Dataset(
        ...     cases=[Case(inputs="What's the weather?")],
        ...     evaluators=[
        ...         ToolUsageEvaluator(
        ...             expected_tools=["get_weather"],
        ...             forbidden_tools=["delete_data"],
        ...         ),
        ...     ],
        ... )

    Returns:
        dict with:
        - called_expected: bool (True if expected tools were called)
        - avoided_forbidden: bool (True if forbidden tools were avoided)
        - tools_called: list of tool names that were called
    """

    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    require_all: bool = True
    check_order: bool = False

    def evaluate(self, ctx: EvaluatorContext[Any, Any]) -> EvaluatorOutput:
        """Evaluate tool usage from span tree."""
        # Get span tree from context (requires logfire configuration)
        span_tree = ctx.span_tree

        # Find all tool-related spans
        tools_called: list[str] = []

        if span_tree is not None:
            # Search for tool call spans
            for node in span_tree:
                # Look for spans that indicate tool calls
                # Common patterns: "tool_call", "execute_tool", tool name directly
                name_lower = node.name.lower()
                if "tool" in name_lower or any(
                    tool.lower() in name_lower for tool in self.expected_tools
                ):
                    # Extract tool name from span
                    tool_name = self._extract_tool_name(node.name, node.attributes)
                    if tool_name:
                        tools_called.append(tool_name)

        # Check expected tools
        called_expected = self._check_expected_tools(tools_called)

        # Check forbidden tools
        avoided_forbidden = self._check_forbidden_tools(tools_called)

        return {
            "called_expected": EvaluationReason(
                value=called_expected,
                reason=self._expected_reason(tools_called),
            ),
            "avoided_forbidden": EvaluationReason(
                value=avoided_forbidden,
                reason=self._forbidden_reason(tools_called),
            ),
            "tools_called": str(tools_called),
        }

    def _extract_tool_name(self, span_name: str, attributes: dict[str, Any]) -> str | None:
        """Extract tool name from span name or attributes."""
        # Check attributes first
        if "tool_name" in attributes:
            return str(attributes["tool_name"])
        if "name" in attributes:
            return str(attributes["name"])

        # Try to extract from span name
        # Common patterns: "tool_call:get_weather", "execute get_weather"
        for sep in [":", " ", "_call_"]:
            if sep in span_name:
                parts = span_name.split(sep)
                # Return the last part that looks like a tool name
                for part in reversed(parts):
                    if part and not part.lower().startswith(("tool", "call", "execute")):
                        return part

        return span_name

    def _check_expected_tools(self, tools_called: list[str]) -> bool:
        """Check if expected tools were called."""
        if not self.expected_tools:
            return True

        tools_called_lower = [t.lower() for t in tools_called]

        if self.require_all:
            # All expected tools must be called
            for expected in self.expected_tools:
                if not any(expected.lower() in t for t in tools_called_lower):
                    return False
            return True
        else:
            # At least one expected tool must be called
            for expected in self.expected_tools:
                if any(expected.lower() in t for t in tools_called_lower):
                    return True
            return False

    def _check_forbidden_tools(self, tools_called: list[str]) -> bool:
        """Check that forbidden tools were NOT called."""
        if not self.forbidden_tools:
            return True

        tools_called_lower = [t.lower() for t in tools_called]

        for forbidden in self.forbidden_tools:
            if any(forbidden.lower() in t for t in tools_called_lower):
                return False
        return True

    def _expected_reason(self, tools_called: list[str]) -> str:
        """Generate reason for expected tools check."""
        if not self.expected_tools:
            return "No expected tools specified"

        found = []
        missing = []
        for expected in self.expected_tools:
            if any(expected.lower() in t.lower() for t in tools_called):
                found.append(expected)
            else:
                missing.append(expected)

        if missing:
            return f"Missing tools: {missing}. Called: {tools_called}"
        return f"All expected tools called: {found}"

    def _forbidden_reason(self, tools_called: list[str]) -> str:
        """Generate reason for forbidden tools check."""
        if not self.forbidden_tools:
            return "No forbidden tools specified"

        violated = []
        for forbidden in self.forbidden_tools:
            if any(forbidden.lower() in t.lower() for t in tools_called):
                violated.append(forbidden)

        if violated:
            return f"Forbidden tools were called: {violated}"
        return "No forbidden tools were called"


@dataclass
class RAGFaithfulness(Evaluator[str, str]):
    """Evaluate if an answer is grounded in the provided context.

    Uses an LLM judge to verify that the generated answer is faithful
    to the retrieved context and doesn't contain hallucinations.

    Args:
        llm_judge: Model to use for judging (e.g., "gpt-4o-mini").
            If None, uses default from environment.
        provider: LLM provider (openai, anthropic, google, etc.).
        context_key: Metadata key containing the context/retrieved docs.
            Default: "context".
        strict: If True, requires exact grounding. If False, allows
            reasonable inferences. Default: False.

    Example:
        >>> from ai_infra.eval.evaluators import RAGFaithfulness
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(
        ...             inputs="What is the refund policy?",
        ...             metadata={"context": "Refunds are available within 30 days."},
        ...         ),
        ...     ],
        ...     evaluators=[RAGFaithfulness(llm_judge="gpt-4o-mini")],
        ... )

    Returns:
        EvaluationReason with:
        - value: float (faithfulness score 0.0-1.0)
        - reason: Explanation from the LLM judge
    """

    llm_judge: str | None = None
    provider: str | None = None
    context_key: str = "context"
    strict: bool = False

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> EvaluatorOutput:
        """Evaluate faithfulness of output to context."""
        # Get context from metadata
        context = None
        if ctx.metadata:
            context = ctx.metadata.get(self.context_key)

        if not context:
            return EvaluationReason(
                value=True,
                reason=f"Skipped: no context found in metadata['{self.context_key}']",
            )

        if not ctx.output:
            return EvaluationReason(
                value=0.0,
                reason="Empty output",
            )

        try:
            # Lazy import to avoid circular dependencies
            from ai_infra import LLM

            llm = LLM(provider=self.provider, model_name=self.llm_judge)

            # Construct the faithfulness prompt
            strictness = "exactly" if self.strict else "reasonably"
            prompt = f"""You are evaluating if an AI-generated answer is {strictness} grounded in the provided context.

Context:
{context}

Question: {ctx.inputs}

Answer: {ctx.output}

Evaluate the faithfulness of the answer to the context. Consider:
1. Is every claim in the answer supported by the context?
2. Does the answer contain any hallucinated information not in the context?
3. Is the answer a reasonable interpretation of the context?

Respond with a JSON object:
{{"score": <float 0.0-1.0>, "reason": "<brief explanation>"}}

Where:
- 1.0 = Fully faithful, all claims grounded in context
- 0.5 = Partially faithful, some claims not grounded
- 0.0 = Unfaithful, contains hallucinations or contradicts context
"""

            response = await llm.ainvoke(prompt)

            # Parse the response
            import json

            try:
                # Try to extract JSON from response
                response_text = str(response)
                # Find JSON in response
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(response_text[start:end])
                    score = float(result.get("score", 0.5))
                    reason = result.get("reason", "No reason provided")
                    return EvaluationReason(value=score, reason=reason)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

            # Fallback: try to extract a score from the response
            return EvaluationReason(
                value=0.5,
                reason=f"Could not parse judge response: {response_text[:200]}",
            )

        except Exception as e:
            return EvaluationReason(
                value=0.5,
                reason=f"Error during faithfulness evaluation: {e}",
            )


@dataclass
class ContainsExpected(Evaluator[str, str]):
    """Check if output contains the expected output text.

    A simple evaluator that checks if the expected_output is contained
    within the output string (case-insensitive by default).

    Args:
        case_sensitive: If True, comparison is case-sensitive. Default: False.

    Example:
        >>> from ai_infra.eval.evaluators import ContainsExpected
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(inputs="capital of France", expected_output="Paris"),
        ...     ],
        ...     evaluators=[ContainsExpected()],
        ... )

    Returns:
        bool: True if expected_output is found in output
    """

    case_sensitive: bool = False

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> bool:
        """Check if output contains expected output."""
        if ctx.expected_output is None:
            return True

        output = str(ctx.output)
        expected = str(ctx.expected_output)

        if not self.case_sensitive:
            output = output.lower()
            expected = expected.lower()

        return expected in output


@dataclass
class LengthInRange(Evaluator[str, str]):
    """Check if output length is within a specified range.

    Useful for ensuring responses are not too short or too long.

    Args:
        min_length: Minimum allowed length. Default: 0.
        max_length: Maximum allowed length. Default: None (no limit).
        count_words: If True, count words instead of characters. Default: False.

    Example:
        >>> from ai_infra.eval.evaluators import LengthInRange
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> dataset = Dataset(
        ...     cases=[Case(inputs="Summarize this", expected_output=None)],
        ...     evaluators=[LengthInRange(min_length=10, max_length=500)],
        ... )

    Returns:
        EvaluationReason with pass/fail and length info
    """

    min_length: int = 0
    max_length: int | None = None
    count_words: bool = False

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> EvaluationReason:
        """Check if output length is in range."""
        output = str(ctx.output)

        if self.count_words:
            length = len(output.split())
            unit = "words"
        else:
            length = len(output)
            unit = "characters"

        if length < self.min_length:
            return EvaluationReason(
                value=False,
                reason=f"Output too short: {length} {unit} < min {self.min_length}",
            )

        if self.max_length is not None and length > self.max_length:
            return EvaluationReason(
                value=False,
                reason=f"Output too long: {length} {unit} > max {self.max_length}",
            )

        return EvaluationReason(
            value=True,
            reason=f"Output length {length} {unit} is within range",
        )
