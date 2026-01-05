"""Evaluation framework for ai-infra.

This module provides integration with pydantic-evals for evaluating AI systems.
Instead of building a custom evaluation framework, we leverage the production-ready
pydantic-evals library and provide thin wrappers that integrate with ai-infra's
Agent and Retriever classes.

Example - Basic agent evaluation:
    >>> from ai_infra import Agent
    >>> from ai_infra.eval import evaluate_agent
    >>> from pydantic_evals import Case, Dataset
    >>>
    >>> agent = Agent(model="gpt-4o-mini")
    >>> dataset = Dataset(
    ...     cases=[
    ...         Case(inputs="What is 2+2?", expected_output="4"),
    ...         Case(inputs="Capital of France?", expected_output="Paris"),
    ...     ],
    ... )
    >>> report = evaluate_agent(agent, dataset)
    >>> report.print()

Example - Retriever evaluation:
    >>> from ai_infra import Retriever
    >>> from ai_infra.eval import evaluate_retriever
    >>>
    >>> retriever = Retriever()
    >>> retriever.add("Paris is the capital of France.")
    >>> retriever.add("Tokyo is the capital of Japan.")
    >>>
    >>> dataset = Dataset(
    ...     cases=[
    ...         Case(inputs="What is the capital of France?", expected_output="Paris"),
    ...     ],
    ... )
    >>> report = evaluate_retriever(retriever, dataset)
    >>> report.print()

For full documentation on pydantic-evals, see:
https://ai.pydantic.dev/evals/
"""

from __future__ import annotations

# Re-export core pydantic-evals classes for convenience
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    IsInstance,
)
from pydantic_evals.reporting import EvaluationReport

# ai-infra custom evaluators (Phase 11.2)
from ai_infra.eval.evaluators import (
    ContainsExpected,
    LengthInRange,
    RAGFaithfulness,
    SemanticSimilarity,
    ToolUsageEvaluator,
)

# ai-infra integration functions
from ai_infra.eval.integration import (
    evaluate_agent,
    evaluate_agent_async,
    evaluate_retriever,
    evaluate_retriever_async,
)

# Logfire integration (Phase 11.4)
from ai_infra.eval.logfire import (
    HasMatchingSpan,
    SpanNode,
    SpanQuery,
    SpanTree,
    check_no_tool_called,
    check_tool_called,
    configure_logfire_evals,
    create_span_query,
    is_logfire_configured,
)

# RAG evaluation helpers (Phase 11.3)
from ai_infra.eval.rag import (
    MRR,
    NDCG,
    DocumentHitRate,
    RetrievalPrecision,
    RetrievalRecall,
    create_retrieval_dataset,
    evaluate_rag_pipeline,
    evaluate_rag_pipeline_async,
)

__all__ = [
    # pydantic-evals core (re-exported for convenience)
    "Case",
    "Dataset",
    "Evaluator",
    "EvaluatorContext",
    "EvaluationReport",
    "IsInstance",
    # ai-infra integration
    "evaluate_agent",
    "evaluate_agent_async",
    "evaluate_retriever",
    "evaluate_retriever_async",
    # ai-infra custom evaluators (Phase 11.2)
    "SemanticSimilarity",
    "ToolUsageEvaluator",
    "RAGFaithfulness",
    "ContainsExpected",
    "LengthInRange",
    # RAG evaluation (Phase 11.3)
    "RetrievalPrecision",
    "RetrievalRecall",
    "DocumentHitRate",
    "MRR",
    "NDCG",
    "create_retrieval_dataset",
    "evaluate_rag_pipeline",
    "evaluate_rag_pipeline_async",
    # Logfire integration (Phase 11.4)
    "configure_logfire_evals",
    "is_logfire_configured",
    "HasMatchingSpan",
    "SpanNode",
    "SpanQuery",
    "SpanTree",
    "create_span_query",
    "check_tool_called",
    "check_no_tool_called",
]
